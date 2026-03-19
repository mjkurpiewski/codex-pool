package main

import (
	"log"
	"sync"
	"time"
)

const (
	bruteForceMaxAttempts = 5
	bruteForceBanDuration = 30 * time.Minute
	bruteForceCleanup     = 10 * time.Minute
)

type attemptRecord struct {
	count    int
	firstAt  time.Time
	bannedAt time.Time
}

// bruteForceTracker tracks failed authentication attempts per IP address
// and bans IPs after too many failures.
type bruteForceTracker struct {
	mu       sync.Mutex
	attempts map[string]*attemptRecord
	stopCh   chan struct{}
}

func newBruteForceTracker() *bruteForceTracker {
	t := &bruteForceTracker{
		attempts: make(map[string]*attemptRecord),
		stopCh:   make(chan struct{}),
	}
	go t.cleanupLoop()
	return t
}

// isBanned returns true if the IP is currently banned.
func (t *bruteForceTracker) isBanned(ip string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	rec, ok := t.attempts[ip]
	if !ok {
		return false
	}
	if rec.bannedAt.IsZero() {
		return false
	}
	if time.Since(rec.bannedAt) > bruteForceBanDuration {
		// Ban expired — clear it.
		delete(t.attempts, ip)
		return false
	}
	return true
}

// recordFailure increments the failure count for an IP. Returns true if
// the IP is now banned (just crossed the threshold).
func (t *bruteForceTracker) recordFailure(ip string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	rec, ok := t.attempts[ip]
	if !ok {
		rec = &attemptRecord{firstAt: time.Now()}
		t.attempts[ip] = rec
	}
	// If already banned, don't re-count.
	if !rec.bannedAt.IsZero() {
		return true
	}
	rec.count++
	if rec.count >= bruteForceMaxAttempts {
		rec.bannedAt = time.Now()
		log.Printf("brute-force: banning IP %s after %d failed auth attempts", ip, rec.count)
		return true
	}
	return false
}

// recordSuccess clears the failure count for an IP.
func (t *bruteForceTracker) recordSuccess(ip string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.attempts, ip)
}

func (t *bruteForceTracker) cleanupLoop() {
	ticker := time.NewTicker(bruteForceCleanup)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			t.cleanup()
		case <-t.stopCh:
			return
		}
	}
}

func (t *bruteForceTracker) cleanup() {
	t.mu.Lock()
	defer t.mu.Unlock()
	now := time.Now()
	for ip, rec := range t.attempts {
		if !rec.bannedAt.IsZero() && now.Sub(rec.bannedAt) > bruteForceBanDuration {
			delete(t.attempts, ip)
		} else if rec.bannedAt.IsZero() && now.Sub(rec.firstAt) > bruteForceBanDuration {
			// Stale non-banned entries.
			delete(t.attempts, ip)
		}
	}
}

func (t *bruteForceTracker) stop() {
	close(t.stopCh)
}

