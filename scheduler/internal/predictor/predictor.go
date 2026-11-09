// predictor.go — wraps the ONNX model for <1ms prompt classification and continuous
// P(Long) probability extraction for SJF queue admission.
//
// Output classes: 0 = Short (<200 tokens), 1 = Medium (200-799), 2 = Long (≥800).
// Output probability: P(Long) in [0.0, 1.0] is used as the continuous priority key.
//
// Thread safety: Predict() is protected by a mutex. Safe for concurrent
// use from multiple HTTP handler goroutines.

package predictor

import (
	"fmt"
	"os"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// Class labels matching train.py.
const (
	Short  = 0
	Medium = 1
	Long   = 2
)

// Prediction contains both discrete classification and continuous probability distribution.
type Prediction struct {
	Class         int        // 0=Short, 1=Medium, 2=Long (argmax)
	PLong         float32    // continuous probability of Long [0.0, 1.0]
	Probabilities [3]float32 // [P(Short), P(Medium), P(Long)]
}

// Predictor wraps an ONNX session for prompt length classification.
type Predictor struct {
	mu           sync.Mutex
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[int64]
	probTensor   *ort.Tensor[float32]
}

// defaultLibPath attempts to locate libonnxruntime if not explicitly configured.
func defaultLibPath() string {
	// 1. Check environment variable
	if p := os.Getenv("ONNX_LIB_PATH"); p != "" {
		return p
	}
	// 2. Platform-specific common locations
	if runtime.GOOS == "darwin" {
		candidates := []string{
			"/opt/homebrew/Caskroom/miniforge/base/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.25.1.dylib",
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"/usr/local/lib/libonnxruntime.dylib",
		}
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				return c
			}
		}
	}
	return ""
}

// New loads the ONNX model and returns a ready Predictor.
// libPath may be empty to use system/auto-detected ONNX Runtime library.
// outputLabel is the name of the class output tensor (typically "label").
func New(modelPath, libPath, outputLabel string) (*Predictor, error) {
	if libPath == "" {
		libPath = defaultLibPath()
	}
	if libPath != "" {
		ort.SetSharedLibraryPath(libPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("ort init: %w", err)
	}

	// Input tensor: shape [1, NFeatures] float32
	inputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, NFeatures))
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	// Output tensor 1: shape [1] int64 — the predicted class label
	outputTensor, err := ort.NewEmptyTensor[int64](ort.NewShape(1))
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}

	// Output tensor 2: shape [1, 3] float32 — the class probabilities
	probTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 3))
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("create prob tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"float_input"},
		[]string{outputLabel, "probabilities"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor, probTensor},
		nil,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		probTensor.Destroy()
		return nil, fmt.Errorf("create ort session: %w", err)
	}

	return &Predictor{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
		probTensor:   probTensor,
	}, nil
}

// Predict extracts features from prompt and returns full Prediction (Class + PLong + Probs).
func (p *Predictor) Predict(prompt string) (Prediction, error) {
	feats := Extract(prompt)
	data := feats.ToSlice()

	p.mu.Lock()
	defer p.mu.Unlock()

	copy(p.inputTensor.GetData(), data)

	if err := p.session.Run(); err != nil {
		return Prediction{Class: Medium, PLong: 0.5}, fmt.Errorf("ort run: %w", err)
	}

	class := int(p.outputTensor.GetData()[0])
	probs := p.probTensor.GetData()

	var pArr [3]float32
	copy(pArr[:], probs)

	return Prediction{
		Class:         class,
		PLong:         probs[2],
		Probabilities: pArr,
	}, nil
}

// PredictClass returns just the discrete class (0, 1, or 2).
func (p *Predictor) PredictClass(prompt string) (int, error) {
	res, err := p.Predict(prompt)
	return res.Class, err
}

// Close releases all ONNX Runtime resources.
func (p *Predictor) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.session.Destroy()
	p.inputTensor.Destroy()
	p.outputTensor.Destroy()
	p.probTensor.Destroy()
	ort.DestroyEnvironment()
}
