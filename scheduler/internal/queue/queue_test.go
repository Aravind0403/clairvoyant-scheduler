package queue

import (
	"context"
	"testing"
	"time"
)

func TestContinuousPLongQueueOrdering(t *testing.T) {
	q := New(10, 10*time.Second)

	// Enqueue 3 requests with different continuous PLong values
	rLong := &Request{
		Ctx:        context.Background(),
		Class:      2,
		PLong:      0.85,
		EnqueuedAt: time.Now(),
		RespChan:   make(chan Result, 1),
	}
	rMed := &Request{
		Ctx:        context.Background(),
		Class:      1,
		PLong:      0.40,
		EnqueuedAt: time.Now(),
		RespChan:   make(chan Result, 1),
	}
	rShort := &Request{
		Ctx:        context.Background(),
		Class:      0,
		PLong:      0.05,
		EnqueuedAt: time.Now(),
		RespChan:   make(chan Result, 1),
	}

	q.Push(rLong)
	q.Push(rMed)
	q.Push(rShort)

	// Must pop Short (0.05) -> Med (0.40) -> Long (0.85)
	p1, _ := q.Pop()
	if p1.PLong != 0.05 {
		t.Fatalf("expected 0.05 first, got %f", p1.PLong)
	}

	p2, _ := q.Pop()
	if p2.PLong != 0.40 {
		t.Fatalf("expected 0.40 second, got %f", p2.PLong)
	}

	p3, _ := q.Pop()
	if p3.PLong != 0.85 {
		t.Fatalf("expected 0.85 third, got %f", p3.PLong)
	}
}

func TestStarvationPromotion(t *testing.T) {
	timeout := 100 * time.Millisecond
	q := New(10, timeout)

	// Enqueue a long request
	rLong := &Request{
		Ctx:        context.Background(),
		Class:      2,
		PLong:      0.90,
		EnqueuedAt: time.Now().Add(-200 * time.Millisecond), // already starved
		RespChan:   make(chan Result, 1),
	}
	// Enqueue a fresh short request
	rShort := &Request{
		Ctx:        context.Background(),
		Class:      0,
		PLong:      0.02,
		EnqueuedAt: time.Now(),
		RespChan:   make(chan Result, 1),
	}

	q.Push(rLong)
	q.Push(rShort)
	q.Reheap()

	// Starved long request must pop first!
	p1, _ := q.Pop()
	if p1 != rLong {
		t.Fatalf("expected starved long request to bubble to top, got pLong=%f", p1.PLong)
	}
	p2, _ := q.Pop()
	if p2 != rShort {
		t.Fatalf("expected short request second, got pLong=%f", p2.PLong)
	}
}

