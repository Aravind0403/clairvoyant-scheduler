package config

import (
	"os"
	"strconv"
	"time"
)

// Config holds all runtime configuration for the scheduler.
// Every field is overridable via environment variable.
type Config struct {
	// Network
	ListenAddr string // LISTEN_ADDR, default :8080
	BackendURL string // BACKEND_URL, default http://localhost:11434

	// Queue
	QueueCapacity     int           // QUEUE_CAPACITY, default 256
	StarvationTimeout time.Duration // STARVATION_TIMEOUT_SEC, default 15

	// ONNX
	ONNXModelPath      string // ONNX_MODEL_PATH, default model/predictor.onnx
	ONNXLibPath        string // ONNX_LIB_PATH, empty = system default
	ONNXOutputLabel    string // ONNX_OUTPUT_LABEL, default "label"
	                          // (verify with export.py output — may be "output_label")
}

func Load() *Config {
	return &Config{
		ListenAddr:        getEnv("LISTEN_ADDR", ":8080"),
		BackendURL:        getEnv("BACKEND_URL", "http://localhost:11434"),
		QueueCapacity:     getEnvInt("QUEUE_CAPACITY", 256),
		StarvationTimeout: time.Duration(getEnvInt("STARVATION_TIMEOUT_SEC", 15)) * time.Second,
		ONNXModelPath:     getEnv("ONNX_MODEL_PATH", "model/predictor.onnx"),
		ONNXLibPath:       getEnv("ONNX_LIB_PATH", ""),
		ONNXOutputLabel:   getEnv("ONNX_OUTPUT_LABEL", "label"),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}
