package main

import (
	"net/http"
	"strconv"
	"strings"
	"time"
)

func parseRateLimitFloat(value string) (float64, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}
	if strings.HasSuffix(value, "%") {
		value = strings.TrimSpace(strings.TrimSuffix(value, "%"))
	}
	f, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0, false
	}
	return f, true
}

func clampRateLimitPercent(value float64) float64 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

// parseRateLimitPercent accepts values in either percent (0-100) or ratio (0-1) form.
func parseRateLimitPercent(value string) (float64, bool) {
	raw, ok := parseRateLimitFloat(value)
	if !ok || raw < 0 {
		return 0, false
	}
	if raw > 1 {
		raw = raw / 100.0
	}
	return clampRateLimitPercent(raw), true
}

func parseRateLimitUsageFromRemainingLimit(headers http.Header, remainingKey, limitKey string) (float64, bool) {
	remaining, ok := parseRateLimitFloat(headers.Get(remainingKey))
	if !ok {
		return 0, false
	}
	limit, ok := parseRateLimitFloat(headers.Get(limitKey))
	if !ok || limit <= 0 {
		return 0, false
	}
	return clampRateLimitPercent((limit - remaining) / limit), true
}

func parseRateLimitReset(value string) (time.Time, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return time.Time{}, false
	}
	if parsed, err := time.Parse(time.RFC3339, value); err == nil {
		return parsed, true
	}
	if parsed, err := strconv.ParseFloat(value, 64); err == nil && parsed > 0 {
		sec := int64(parsed)
		// Handle ms timestamps if they are clearly in that domain.
		if sec > 1e12 {
			sec = int64(parsed / 1000)
		}
		return time.Unix(sec, 0), true
	}
	return time.Time{}, false
}
