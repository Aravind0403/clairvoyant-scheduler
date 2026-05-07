// main.go — Clairvoyant Scheduler entry point.
//
// Wires together:
//   config     → env-driven configuration
//   predictor  → ONNX model (<1ms per prompt)
//   queue      → SJF priority queue with starvation prevention
//   proxy      → HTTP handler (intercepts /v1/chat/completions)
//   dispatcher → single-dispatch goroutine to backend
//   aging      → reheap goroutine (default 1s tick)
//
// Usage:
//   ONNX_MODEL_PATH=../model/predictor.onnx \
//   BACKEND_URL=http://localhost:11434 \
//   go run ./cmd/main.go

package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/aravindsundaresan/clairvoyant-scheduler/internal/config"
	"github.com/aravindsundaresan/clairvoyant-scheduler/internal/predictor"
	"github.com/aravindsundaresan/clairvoyant-scheduler/internal/proxy"
	"github.com/aravindsundaresan/clairvoyant-scheduler/internal/queue"
)

func main() {
	cfg := config.Load()

	// ── predictor ─────────────────────────────────────────────────────────────
	log.Printf("loading ONNX model: %s", cfg.ONNXModelPath)
	pred, err := predictor.New(cfg.ONNXModelPath, cfg.ONNXLibPath, cfg.ONNXOutputLabel)
	if err != nil {
		log.Fatalf("predictor init failed: %v\n"+
			"  → check ONNX_MODEL_PATH and that libonnxruntime is installed\n"+
			"  → if output tensor name is wrong, set ONNX_OUTPUT_LABEL (check export.py output)", err)
	}
	defer pred.Close()
	log.Printf("predictor ready (output label=%q)", cfg.ONNXOutputLabel)

	// ── queue ─────────────────────────────────────────────────────────────────
	q := queue.New(cfg.QueueCapacity, cfg.StarvationTimeout)
	log.Printf("queue: capacity=%d starvation_timeout=%s", cfg.QueueCapacity, cfg.StarvationTimeout)

	// ── dispatcher ────────────────────────────────────────────────────────────
	dispatcher := proxy.NewDispatcher(cfg.BackendURL, q)
	go dispatcher.Run()

	// ── aging monitor (1s tick) ───────────────────────────────────────────────
	aging := proxy.NewAgingMonitor(q, 1*time.Second)
	go aging.Run()

	// ── HTTP server ───────────────────────────────────────────────────────────
	handler := proxy.NewHandler(cfg.BackendURL, pred, q)
	server := &http.Server{
		Addr:         cfg.ListenAddr,
		Handler:      handler,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 10 * time.Minute, // LLM responses can be slow
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		log.Printf("clairvoyant scheduler listening on %s → %s", cfg.ListenAddr, cfg.BackendURL)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	// ── graceful shutdown ─────────────────────────────────────────────────────
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop

	log.Println("shutting down...")
	aging.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		log.Printf("server shutdown error: %v", err)
	}

	q.Close()
	log.Println("stopped")
}
