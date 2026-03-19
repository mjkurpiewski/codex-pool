package main

import (
	"log"
	"strings"
	"sync"
)

// modelAliases manages model name aliases. Thread-safe for hot-reload.
type modelAliases struct {
	mu      sync.RWMutex
	aliases map[string]string // short name -> full upstream model name
}

func newModelAliases(cfg map[string]string) *modelAliases {
	m := &modelAliases{aliases: make(map[string]string)}
	if cfg != nil {
		for k, v := range cfg {
			m.aliases[strings.ToLower(k)] = v
		}
	}
	return m
}

// resolve returns the upstream model name for a given alias, or the
// original name if no alias is defined. The second return value indicates
// whether an alias was applied.
func (m *modelAliases) resolve(model string) (string, bool) {
	if m == nil || model == "" {
		return model, false
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if target, ok := m.aliases[strings.ToLower(model)]; ok {
		return target, true
	}
	return model, false
}

// reload replaces the alias map (used by hot-reload).
func (m *modelAliases) reload(cfg map[string]string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.aliases = make(map[string]string, len(cfg))
	for k, v := range cfg {
		m.aliases[strings.ToLower(k)] = v
	}
}

// applyModelAlias resolves the model alias and rewrites the body if needed.
// Uses the existing rewriteModelInBody from main.go.
func applyModelAlias(aliases *modelAliases, model string, body []byte, debug bool, reqID string) (string, []byte) {
	resolved, aliased := aliases.resolve(model)
	if !aliased {
		return model, body
	}
	if rewritten := rewriteModelInBody(body, resolved); rewritten != nil {
		if debug {
			log.Printf("[%s] model alias: %s -> %s", reqID, model, resolved)
		}
		return resolved, rewritten
	}
	// Body rewrite failed, but still use the resolved name for routing.
	if debug {
		log.Printf("[%s] model alias: %s -> %s (body rewrite failed, routing only)", reqID, model, resolved)
	}
	return resolved, body
}
