// predictor.go — wraps the ONNX model for <1ms prompt classification.
//
// Output: 0 = Short (<200 tokens), 1 = Medium (200-799), 2 = Long (≥800).
//
// NOTE: The ONNX output tensor name defaults to "label" (standard for
// onnxmltools XGBoost exports). If inference fails, verify the actual
// name from export.py's printed output and set ONNX_OUTPUT_LABEL env var.
//
// Thread safety: Predict() is protected by a mutex. Safe for concurrent
// use from multiple HTTP handler goroutines.

package predictor

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// Class labels matching train.py.
const (
	Short  = 0
	Medium = 1
	Long   = 2
)

// Predictor wraps an ONNX session for prompt length classification.
type Predictor struct {
	mu           sync.Mutex
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[int64]
}

// New loads the ONNX model and returns a ready Predictor.
// libPath may be empty to use the system ONNX Runtime library.
// outputLabel is the name of the class output tensor (typically "label").
func New(modelPath, libPath, outputLabel string) (*Predictor, error) {
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

	// Output tensor: shape [1] int64 — the predicted class label
	outputTensor, err := ort.NewEmptyTensor[int64](ort.NewShape(1))
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"float_input"},
		[]string{outputLabel},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		nil,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("create ort session: %w", err)
	}

	return &Predictor{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
	}, nil
}

// Predict extracts features from prompt and returns 0 (Short), 1 (Medium), or 2 (Long).
func (p *Predictor) Predict(prompt string) (int, error) {
	feats := Extract(prompt)
	data := feats.ToSlice()

	p.mu.Lock()
	defer p.mu.Unlock()

	copy(p.inputTensor.GetData(), data)

	if err := p.session.Run(); err != nil {
		return 0, fmt.Errorf("ort run: %w", err)
	}

	return int(p.outputTensor.GetData()[0]), nil
}

// Close releases all ONNX Runtime resources.
func (p *Predictor) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.session.Destroy()
	p.inputTensor.Destroy()
	p.outputTensor.Destroy()
	ort.DestroyEnvironment()
}
