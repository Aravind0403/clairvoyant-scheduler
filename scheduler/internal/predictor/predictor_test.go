package predictor

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPredictorProbabilities(t *testing.T) {
	wd, _ := os.Getwd()
	repoRoot := filepath.Clean(filepath.Join(wd, "..", "..", ".."))
	modelPath := filepath.Join(repoRoot, "model", "predictor.onnx")

	pred, err := New(modelPath, "", "label")
	if err != nil {
		t.Fatalf("failed to init predictor: %v", err)
	}
	defer pred.Close()

	// Short prompt
	resShort, err := pred.Predict("What is the capital of France?")
	if err != nil {
		t.Fatalf("predict short failed: %v", err)
	}
	t.Logf("Short prompt -> Class: %d, PLong: %f, Probs: %v", resShort.Class, resShort.PLong, resShort.Probabilities)
	if resShort.Class != Short {
		t.Errorf("expected Short class, got %d", resShort.Class)
	}
	if resShort.PLong > 0.3 {
		t.Errorf("expected low PLong for short query, got %f", resShort.PLong)
	}

	// Long prompt
	resLong, err := pred.Predict("can you help me make a tic tac toe game using react?")
	if err != nil {
		t.Fatalf("predict long failed: %v", err)
	}
	t.Logf("Long prompt -> Class: %d, PLong: %f, Probs: %v", resLong.Class, resLong.PLong, resLong.Probabilities)
	if resLong.Class != Long {
		t.Errorf("expected Long class, got %d", resLong.Class)
	}
	if resLong.PLong < resShort.PLong {
		t.Errorf("expected PLong(Long) > PLong(Short), got %f <= %f", resLong.PLong, resShort.PLong)
	}
}
