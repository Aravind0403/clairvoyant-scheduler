// queue.go — thread-safe SJF priority queue with continuous P(Long) ordering
// and starvation prevention.
//
// Ordering:
//   Primary:   effective_priority (ascending P(Long) in [0.0, 1.0])
//   Secondary: enqueue_time (FIFO for requests with equal priority)
//
// Starvation prevention:
//   If a request has waited ≥ StarvationTimeout, its effective priority
//   is promoted to negative values (< 0.0), guaranteeing that starved requests
//   bubble to the top. The oldest starving request gets the lowest value,
//   preserving FIFO among starved requests.
//
// Queue full policy: Push() returns false → caller sends HTTP 429.

package queue

import (
	"container/heap"
	"context"
	"sync"
	"time"
)

// Result carries the backend response back to the waiting HTTP handler.
type Result struct {
	StatusCode int
	Headers    map[string][]string
	Body       []byte
	Err        error
}

// Request is one inference job waiting in the priority queue.
type Request struct {
	Ctx        context.Context // client HTTP request context for cancellation
	Body       []byte
	Class      int             // predictor output: 0=Short, 1=Medium, 2=Long (argmax)
	PLong      float32         // continuous scheduling key in [0.0, 1.0]
	EnqueuedAt time.Time
	RespChan   chan Result

	index int // maintained by heap.Interface — do not set externally
}

// effectivePriority returns the scheduling priority, accounting for aging.
// If a request has waited ≥ timeout, its priority becomes negative (< 0.0),
// bubbling it ahead of all unstarved requests (with oldest starved requests first).
// Otherwise, it returns PLong in [0.0, 1.0].
func (r *Request) effectivePriority(timeout time.Duration) float32 {
	waited := time.Since(r.EnqueuedAt)
	if waited >= timeout {
		// Negative priority proportional to how long it has been starving.
		// Older starved requests get more negative values, popping first.
		return -1.0 - float32(waited.Seconds())
	}
	return r.PLong
}

// ── internal heap ────────────────────────────────────────────────────────────

type minHeap struct {
	items   []*Request
	timeout time.Duration
}

func (h minHeap) Len() int { return len(h.items) }

func (h minHeap) Less(i, j int) bool {
	pi := h.items[i].effectivePriority(h.timeout)
	pj := h.items[j].effectivePriority(h.timeout)
	if pi != pj {
		return pi < pj
	}
	return h.items[i].EnqueuedAt.Before(h.items[j].EnqueuedAt)
}

func (h minHeap) Swap(i, j int) {
	h.items[i], h.items[j] = h.items[j], h.items[i]
	h.items[i].index = i
	h.items[j].index = j
}

func (h *minHeap) Push(x any) {
	r := x.(*Request)
	r.index = len(h.items)
	h.items = append(h.items, r)
}

func (h *minHeap) Pop() any {
	old := h.items
	n := len(old)
	r := old[n-1]
	old[n-1] = nil
	h.items = old[:n-1]
	r.index = -1
	return r
}

// ── public Queue ─────────────────────────────────────────────────────────────

// Queue is a thread-safe SJF priority queue.
type Queue struct {
	mu     sync.Mutex
	h      minHeap
	cap    int
	ready  chan struct{} // signals dispatcher: at least one item available
	closed bool
}

// New creates a queue with the given capacity and starvation timeout.
func New(capacity int, starvationTimeout time.Duration) *Queue {
	q := &Queue{
		cap:   capacity,
		ready: make(chan struct{}, 1),
		h: minHeap{
			items:   make([]*Request, 0, capacity),
			timeout: starvationTimeout,
		},
	}
	heap.Init(&q.h)
	return q
}

// Push enqueues a request. Returns false if the queue is closed or full (caller → 429).
func (q *Queue) Push(r *Request) bool {
	q.mu.Lock()
	defer q.mu.Unlock()

	if q.closed || len(q.h.items) >= q.cap {
		return false
	}
	heap.Push(&q.h, r)

	// Non-blocking wake-up signal to dispatcher.
	select {
	case q.ready <- struct{}{}:
	default:
	}
	return true
}

// Pop removes and returns the highest-priority request.
// Blocks until an item is available or the queue is closed.
// Returns (nil, false) when the queue is closed and empty.
func (q *Queue) Pop() (*Request, bool) {
	for {
		q.mu.Lock()
		if len(q.h.items) > 0 {
			r := heap.Pop(&q.h).(*Request)
			q.mu.Unlock()
			return r, true
		}
		q.mu.Unlock()

		_, ok := <-q.ready
		if !ok {
			return nil, false // queue closed
		}
	}
}

// Reheap re-evaluates all priority comparisons — call periodically from
// the AgingMonitor so that requests promoted by starvation bubble to the top.
// Returns the number of requests currently in starved state (waited ≥ timeout).
func (q *Queue) Reheap() int {
	q.mu.Lock()
	defer q.mu.Unlock()

	promoted := 0
	for _, r := range q.h.items {
		if time.Since(r.EnqueuedAt) >= q.h.timeout {
			promoted++
		}
	}
	heap.Init(&q.h)
	return promoted
}

// Close unblocks any waiting Pop() calls and signals the dispatcher to stop.
func (q *Queue) Close() {
	q.mu.Lock()
	defer q.mu.Unlock()
	if !q.closed {
		q.closed = true
		close(q.ready)
	}
}

// Len returns the current number of queued requests.
func (q *Queue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.h.items)
}
