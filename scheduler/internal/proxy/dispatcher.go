// dispatcher.go — two goroutines:
//
//   Dispatcher   pops from the SJF queue and forwards to the LLM backend.
//   AgingMonitor reheaps the queue every second so that requests promoted
//                by the starvation timeout bubble to the top.

package proxy

import (
	"bytes"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/aravindsundaresan/clairvoyant-scheduler/internal/queue"
)

// ── Dispatcher ───────────────────────────────────────────────────────────────

// Dispatcher pops requests from the SJF queue and forwards them one at a time
// to the backend (single dispatch — Ollama is single-threaded).
type Dispatcher struct {
	backendURL string
	q          *queue.Queue
	client     *http.Client
}

func NewDispatcher(backendURL string, q *queue.Queue) *Dispatcher {
	return &Dispatcher{
		backendURL: backendURL,
		q:          q,
		client:     &http.Client{Timeout: 5 * time.Minute},
	}
}

// Run blocks until the queue is closed. Call as a goroutine.
func (d *Dispatcher) Run() {
	log.Println("dispatcher: started")
	for {
		req, ok := d.q.Pop()
		if !ok {
			log.Println("dispatcher: queue closed, stopping")
			return
		}

		waited := time.Since(req.EnqueuedAt)
		log.Printf("dispatcher: dequeued class=%d waited=%.1fms qlen=%d",
			req.Class, float64(waited.Microseconds())/1000.0, d.q.Len())

		req.RespChan <- d.forward(req)
	}
}

// forward sends the request body to the backend and returns the result.
func (d *Dispatcher) forward(req *queue.Request) queue.Result {
	backendReq, err := http.NewRequest(
		http.MethodPost,
		d.backendURL+"/v1/chat/completions",
		bytes.NewReader(req.Body),
	)
	if err != nil {
		return queue.Result{Err: err}
	}
	backendReq.Header.Set("Content-Type", "application/json")

	resp, err := d.client.Do(backendReq)
	if err != nil {
		return queue.Result{Err: err}
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return queue.Result{Err: err}
	}

	headers := make(map[string][]string, len(resp.Header))
	for k, vs := range resp.Header {
		headers[k] = vs
	}

	return queue.Result{
		StatusCode: resp.StatusCode,
		Headers:    headers,
		Body:       body,
	}
}

// ── AgingMonitor ─────────────────────────────────────────────────────────────

// AgingMonitor reheaps the priority queue every interval so that requests
// whose effective priority changed (due to starvation timeout expiry) are
// correctly ordered before the next dispatch.
type AgingMonitor struct {
	q        *queue.Queue
	interval time.Duration
	stop     chan struct{}
}

func NewAgingMonitor(q *queue.Queue, interval time.Duration) *AgingMonitor {
	return &AgingMonitor{
		q:        q,
		interval: interval,
		stop:     make(chan struct{}),
	}
}

// Run ticks on interval and reheaps. Call as a goroutine.
// Only logs when at least one request was actually promoted by aging.
func (m *AgingMonitor) Run() {
	ticker := time.NewTicker(m.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			if promoted := m.q.Reheap(); promoted > 0 {
				log.Printf("aging monitor: promoted %d request(s) to priority-0", promoted)
			}
		case <-m.stop:
			return
		}
	}
}

// Stop shuts down the aging monitor.
func (m *AgingMonitor) Stop() {
	close(m.stop)
}
