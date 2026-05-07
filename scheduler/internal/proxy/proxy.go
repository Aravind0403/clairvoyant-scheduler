// proxy.go — HTTP handler that intercepts /v1/chat/completions,
// predicts output length, and enqueues for SJF dispatch.
//
// All other paths are forwarded to the backend unchanged.
// Queue-full condition returns HTTP 429.

package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/aravindsundaresan/clairvoyant-scheduler/internal/predictor"
	"github.com/aravindsundaresan/clairvoyant-scheduler/internal/queue"
)

// Handler intercepts inference requests and queues them for SJF dispatch.
type Handler struct {
	backendURL string
	pred       *predictor.Predictor
	q          *queue.Queue
	client     *http.Client
}

func NewHandler(backendURL string, pred *predictor.Predictor, q *queue.Queue) *Handler {
	return &Handler{
		backendURL: backendURL,
		pred:       pred,
		q:          q,
		client:     &http.Client{Timeout: 5 * time.Minute},
	}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Only intercept POST /v1/chat/completions; proxy everything else.
	if r.URL.Path != "/v1/chat/completions" || r.Method != http.MethodPost {
		h.passThrough(w, r)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, `{"error":"failed to read request body"}`, http.StatusBadRequest)
		return
	}

	class := h.classify(body)

	respChan := make(chan queue.Result, 1)
	req := &queue.Request{
		Body:       body,
		Class:      class,
		EnqueuedAt: time.Now(),
		RespChan:   respChan,
	}

	if !h.q.Push(req) {
		log.Printf("queue full (depth=%d) — returning 429", h.q.Len())
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", "2")
		http.Error(w, `{"error":"scheduler queue full, retry later"}`, http.StatusTooManyRequests)
		return
	}

	result := <-respChan
	if result.Err != nil {
		http.Error(w,
			fmt.Sprintf(`{"error":"backend error: %v"}`, result.Err),
			http.StatusBadGateway)
		return
	}

	for k, vs := range result.Headers {
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(result.StatusCode)
	_, _ = w.Write(result.Body)
}

// classify predicts the output length class (0/1/2) for a raw request body.
// Falls back to Medium (1) on any error so the request is never dropped.
func (h *Handler) classify(body []byte) int {
	prompt, err := extractLastUserMessage(body)
	if err != nil {
		log.Printf("warn: prompt extraction failed (%v) — defaulting to Medium", err)
		return predictor.Medium
	}

	class, err := h.pred.Predict(prompt)
	if err != nil {
		log.Printf("warn: prediction failed (%v) — defaulting to Medium", err)
		return predictor.Medium
	}

	classNames := [3]string{"Short", "Medium", "Long"}
	log.Printf("predict: class=%d (%s)  prompt=%q", class, classNames[class], truncate(prompt, 60))
	return class
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// extractLastUserMessage pulls the last "user" role content from an
// OpenAI-compatible chat/completions request body.
func extractLastUserMessage(body []byte) (string, error) {
	var payload struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", fmt.Errorf("json decode: %w", err)
	}
	for i := len(payload.Messages) - 1; i >= 0; i-- {
		if payload.Messages[i].Role == "user" {
			return payload.Messages[i].Content, nil
		}
	}
	return "", fmt.Errorf("no user message in payload")
}

// passThrough forwards a request to the backend without modification.
func (h *Handler) passThrough(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)

	req, err := http.NewRequest(r.Method, h.backendURL+r.URL.RequestURI(), bytes.NewReader(body))
	if err != nil {
		http.Error(w, `{"error":"proxy error"}`, http.StatusInternalServerError)
		return
	}
	for k, vs := range r.Header {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}

	resp, err := h.client.Do(req)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"%v"}`, err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	for k, vs := range resp.Header {
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(respBody)
}
