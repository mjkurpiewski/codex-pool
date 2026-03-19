package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"
)

// responsesToChatCompletionsWriter intercepts upstream Responses API SSE events
// and translates them to OpenAI Chat Completions streaming format.
type responsesToChatCompletionsWriter struct {
	w        io.Writer
	buf      []byte
	callback func([]byte) // called with original event data for usage parsing
	debug    bool
	reqID    string

	// State tracking
	id             string
	model          string
	started        bool
	toolCallIndex  int
	inputTokens    int64
	outputTokens   int64
}

func (rw *responsesToChatCompletionsWriter) Write(p []byte) (int, error) {
	origLen := len(p)
	rw.buf = append(rw.buf, p...)
	rw.scanAndTranslate()
	return origLen, nil
}

func (rw *responsesToChatCompletionsWriter) scanAndTranslate() {
	for {
		idx := bytes.Index(rw.buf, []byte("\n\n"))
		advance := 2
		if idx < 0 {
			idx = bytes.Index(rw.buf, []byte("\r\n\r\n"))
			advance = 4
			if idx < 0 {
				if len(rw.buf) > 1024*1024 {
					rw.buf = rw.buf[len(rw.buf)-512*1024:]
				}
				return
			}
		}

		event := rw.buf[:idx]
		rw.buf = rw.buf[idx+advance:]
		rw.processEvent(event)
	}
}

func (rw *responsesToChatCompletionsWriter) processEvent(event []byte) {
	var eventType string
	var data []byte

	for _, line := range bytes.Split(event, []byte("\n")) {
		line = bytes.TrimRight(line, "\r")
		if bytes.HasPrefix(line, []byte("event:")) {
			eventType = string(bytes.TrimSpace(line[6:]))
		} else if bytes.HasPrefix(line, []byte("event: ")) {
			eventType = string(bytes.TrimSpace(line[7:]))
		} else if bytes.HasPrefix(line, []byte("data: ")) {
			data = bytes.TrimSpace(line[6:])
		} else if bytes.HasPrefix(line, []byte("data:")) {
			data = bytes.TrimSpace(line[5:])
		}
	}

	// Forward original data to usage callback
	if len(data) > 0 && rw.callback != nil && !bytes.Equal(data, []byte("[DONE]")) {
		rw.callback(data)
	}

	if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
		return
	}

	var obj map[string]any
	if err := json.Unmarshal(data, &obj); err != nil {
		return
	}

	if eventType == "" {
		if t, ok := obj["type"].(string); ok {
			eventType = t
		}
	}

	switch eventType {
	case "response.created":
		resp, _ := obj["response"].(map[string]any)
		if resp != nil {
			if id, ok := resp["id"].(string); ok {
				rw.id = id
			}
			if m, ok := resp["model"].(string); ok {
				rw.model = m
			}
		}
		// Emit initial role chunk
		rw.started = true
		rw.emitChunk(map[string]any{"role": "assistant", "content": ""}, "", nil)

	case "response.output_item.added":
		item, _ := obj["item"].(map[string]any)
		if item == nil {
			return
		}
		itemType, _ := item["type"].(string)
		switch itemType {
		case "function_call":
			callID, _ := item["call_id"].(string)
			name, _ := item["name"].(string)
			rw.emitChunk(map[string]any{
				"tool_calls": []any{
					map[string]any{
						"index": rw.toolCallIndex,
						"id":    callID,
						"type":  "function",
						"function": map[string]any{
							"name":      name,
							"arguments": "",
						},
					},
				},
			}, "", nil)
			rw.toolCallIndex++
		}

	case "response.output_text.delta":
		delta, _ := obj["delta"].(string)
		if delta != "" {
			rw.emitChunk(map[string]any{"content": delta}, "", nil)
		}

	case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
		delta, _ := obj["delta"].(string)
		if delta != "" {
			rw.emitChunk(map[string]any{"reasoning_content": delta}, "", nil)
		}

	case "response.function_call_arguments.delta":
		delta, _ := obj["delta"].(string)
		if delta != "" {
			idx := rw.toolCallIndex - 1
			if idx < 0 {
				idx = 0
			}
			rw.emitChunk(map[string]any{
				"tool_calls": []any{
					map[string]any{
						"index": idx,
						"function": map[string]any{
							"arguments": delta,
						},
					},
				},
			}, "", nil)
		}

	case "response.output_item.done":
		// No action needed for chat completions format

	case "response.completed":
		resp, _ := obj["response"].(map[string]any)
		if resp != nil {
			if usage, ok := resp["usage"].(map[string]any); ok {
				rw.inputTokens = toInt64(usage["input_tokens"])
				rw.outputTokens = toInt64(usage["output_tokens"])
			}
		}

		// Determine finish reason
		finishReason := "stop"
		if resp != nil {
			if status, ok := resp["status"].(string); ok {
				switch status {
				case "completed":
					finishReason = "stop"
				case "incomplete":
					if reason, ok := resp["incomplete_details"].(map[string]any); ok {
						if r, _ := reason["reason"].(string); r == "max_output_tokens" {
							finishReason = "length"
						}
					}
				}
			}
		}
		if rw.toolCallIndex > 0 {
			finishReason = "tool_calls"
		}

		usage := map[string]any{
			"prompt_tokens":     rw.inputTokens,
			"completion_tokens": rw.outputTokens,
			"total_tokens":      rw.inputTokens + rw.outputTokens,
		}
		rw.emitChunk(map[string]any{}, finishReason, usage)
		rw.writeRaw("data: [DONE]\n\n")

	case "response.failed":
		// Emit error as a final chunk
		resp, _ := obj["response"].(map[string]any)
		errMsg := "response failed"
		if resp != nil {
			if e, ok := resp["error"].(map[string]any); ok {
				if m, ok := e["message"].(string); ok && m != "" {
					errMsg = m
				}
			}
		}
		rw.emitChunk(map[string]any{"content": "[Error: " + errMsg + "]"}, "stop", nil)
		rw.writeRaw("data: [DONE]\n\n")

	case "response.output_text.done", "response.content_part.done",
		"response.content_part.added", "response.reasoning_text.done",
		"response.reasoning_summary_text.done", "response.function_call_arguments.done",
		"response.in_progress":
		// Informational events, no action needed

	default:
		if rw.debug {
			log.Printf("[%s] responses->chat: unhandled event type: %s", rw.reqID, eventType)
		}
	}
}

func (rw *responsesToChatCompletionsWriter) emitChunk(delta map[string]any, finishReason string, usage map[string]any) {
	id := rw.id
	if id == "" {
		id = "chatcmpl-translated"
	}
	model := rw.model
	if model == "" {
		model = "unknown"
	}

	choiceObj := map[string]any{
		"index": 0,
		"delta": delta,
	}
	if finishReason != "" {
		choiceObj["finish_reason"] = finishReason
	} else {
		choiceObj["finish_reason"] = nil
	}

	chunk := map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"model":   model,
		"choices": []any{choiceObj},
	}

	if usage != nil {
		chunk["usage"] = usage
	}

	b, err := json.Marshal(chunk)
	if err != nil {
		return
	}
	rw.writeRaw(fmt.Sprintf("data: %s\n\n", string(b)))
}

func (rw *responsesToChatCompletionsWriter) writeRaw(s string) {
	if _, err := rw.w.Write([]byte(s)); err != nil {
		if rw.debug {
			log.Printf("[%s] responses->chat write error: %v", rw.reqID, err)
		}
	}
}

// responsesToChatCompletionsBufferingWriter is like responsesToChatCompletionsWriter
// but buffers all SSE events and produces a single non-streaming JSON response.
// Used when the client sends stream:false but Codex backend requires streaming.
type responsesToChatCompletionsBufferingWriter struct {
	buf      []byte
	callback func([]byte) // usage callback
	debug    bool
	reqID    string

	// State accumulated from SSE events
	id            string
	model         string
	contentText   string
	toolCalls     []any
	toolCallIndex int
	inputTokens   int64
	outputTokens  int64
	finishReason  string
	errMsg        string
}

func (bw *responsesToChatCompletionsBufferingWriter) Write(p []byte) (int, error) {
	origLen := len(p)
	bw.buf = append(bw.buf, p...)
	bw.scanEvents()
	return origLen, nil
}

func (bw *responsesToChatCompletionsBufferingWriter) scanEvents() {
	for {
		idx := bytes.Index(bw.buf, []byte("\n\n"))
		advance := 2
		if idx < 0 {
			idx = bytes.Index(bw.buf, []byte("\r\n\r\n"))
			advance = 4
			if idx < 0 {
				return
			}
		}
		event := bw.buf[:idx]
		bw.buf = bw.buf[idx+advance:]
		bw.processEvent(event)
	}
}

func (bw *responsesToChatCompletionsBufferingWriter) processEvent(event []byte) {
	var eventType string
	var data []byte

	for _, line := range bytes.Split(event, []byte("\n")) {
		line = bytes.TrimRight(line, "\r")
		if bytes.HasPrefix(line, []byte("event:")) {
			eventType = string(bytes.TrimSpace(line[6:]))
		} else if bytes.HasPrefix(line, []byte("event: ")) {
			eventType = string(bytes.TrimSpace(line[7:]))
		} else if bytes.HasPrefix(line, []byte("data: ")) {
			data = bytes.TrimSpace(line[6:])
		} else if bytes.HasPrefix(line, []byte("data:")) {
			data = bytes.TrimSpace(line[5:])
		}
	}

	if len(data) > 0 && bw.callback != nil && !bytes.Equal(data, []byte("[DONE]")) {
		bw.callback(data)
	}

	if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
		return
	}

	var obj map[string]any
	if err := json.Unmarshal(data, &obj); err != nil {
		return
	}

	if eventType == "" {
		if t, ok := obj["type"].(string); ok {
			eventType = t
		}
	}

	switch eventType {
	case "response.created":
		resp, _ := obj["response"].(map[string]any)
		if resp != nil {
			if id, ok := resp["id"].(string); ok {
				bw.id = id
			}
			if m, ok := resp["model"].(string); ok {
				bw.model = m
			}
		}

	case "response.output_text.delta":
		delta, _ := obj["delta"].(string)
		bw.contentText += delta

	case "response.output_item.added":
		item, _ := obj["item"].(map[string]any)
		if item == nil {
			return
		}
		if itemType, _ := item["type"].(string); itemType == "function_call" {
			callID, _ := item["call_id"].(string)
			name, _ := item["name"].(string)
			bw.toolCalls = append(bw.toolCalls, map[string]any{
				"id":    callID,
				"type":  "function",
				"index": bw.toolCallIndex,
				"function": map[string]any{
					"name":      name,
					"arguments": "",
				},
			})
			bw.toolCallIndex++
		}

	case "response.function_call_arguments.delta":
		delta, _ := obj["delta"].(string)
		if delta != "" && len(bw.toolCalls) > 0 {
			tc := bw.toolCalls[len(bw.toolCalls)-1].(map[string]any)
			fn := tc["function"].(map[string]any)
			fn["arguments"] = fn["arguments"].(string) + delta
		}

	case "response.completed":
		resp, _ := obj["response"].(map[string]any)
		if resp != nil {
			if usage, ok := resp["usage"].(map[string]any); ok {
				bw.inputTokens = toInt64(usage["input_tokens"])
				bw.outputTokens = toInt64(usage["output_tokens"])
			}
		}
		bw.finishReason = "stop"
		if resp != nil {
			if status, ok := resp["status"].(string); ok {
				switch status {
				case "incomplete":
					if reason, ok := resp["incomplete_details"].(map[string]any); ok {
						if r, _ := reason["reason"].(string); r == "max_output_tokens" {
							bw.finishReason = "length"
						}
					}
				}
			}
		}
		if bw.toolCallIndex > 0 {
			bw.finishReason = "tool_calls"
		}

	case "response.failed":
		resp, _ := obj["response"].(map[string]any)
		bw.errMsg = "response failed"
		if resp != nil {
			if e, ok := resp["error"].(map[string]any); ok {
				if m, ok := e["message"].(string); ok && m != "" {
					bw.errMsg = m
				}
			}
		}
		bw.finishReason = "stop"
	}
}

// Result returns the assembled non-streaming Chat Completions JSON response.
func (bw *responsesToChatCompletionsBufferingWriter) Result() []byte {
	id := bw.id
	if id == "" {
		id = "chatcmpl-translated"
	}
	model := bw.model
	if model == "" {
		model = "unknown"
	}

	content := bw.contentText
	if bw.errMsg != "" {
		content = "[Error: " + bw.errMsg + "]"
	}

	message := map[string]any{
		"role":    "assistant",
		"content": content,
	}
	if len(bw.toolCalls) > 0 {
		message["tool_calls"] = bw.toolCalls
	}

	finishReason := bw.finishReason
	if finishReason == "" {
		finishReason = "stop"
	}

	out := map[string]any{
		"id":     id,
		"object": "chat.completion",
		"model":  model,
		"choices": []any{
			map[string]any{
				"index":         0,
				"message":       message,
				"finish_reason": finishReason,
			},
		},
	}

	if bw.inputTokens > 0 || bw.outputTokens > 0 {
		out["usage"] = map[string]any{
			"prompt_tokens":     bw.inputTokens,
			"completion_tokens": bw.outputTokens,
			"total_tokens":      bw.inputTokens + bw.outputTokens,
		}
	}

	b, _ := json.Marshal(out)
	return b
}

// claudeToResponsesWriter translates Claude Messages API SSE events
// to OpenAI Responses API SSE events. Used when Codex CLI sends to /responses
// with a Claude model.
type claudeToResponsesWriter struct {
	w        io.Writer
	buf      []byte
	callback func([]byte)
	debug    bool
	reqID    string

	// State
	id               string
	model            string
	started          bool
	outputIndex      int
	contentIndex     int
	currentBlockType string // "text", "thinking", "tool_use"
	currentToolID    string
	currentToolName  string
	accumulatedText  string
	accumulatedArgs  string
	inputTokens      int64
	outputTokens     int64
	stopReason       string
	sentMessageItem  bool // whether we've emitted the message output_item.added
}

func (cw *claudeToResponsesWriter) Write(p []byte) (int, error) {
	origLen := len(p)
	cw.buf = append(cw.buf, p...)
	cw.scanAndTranslate()
	return origLen, nil
}

func (cw *claudeToResponsesWriter) scanAndTranslate() {
	for {
		idx := bytes.Index(cw.buf, []byte("\n\n"))
		advance := 2
		if idx < 0 {
			idx = bytes.Index(cw.buf, []byte("\r\n\r\n"))
			advance = 4
			if idx < 0 {
				if len(cw.buf) > 1024*1024 {
					cw.buf = cw.buf[len(cw.buf)-512*1024:]
				}
				return
			}
		}
		event := cw.buf[:idx]
		cw.buf = cw.buf[idx+advance:]
		cw.processEvent(event)
	}
}

func (cw *claudeToResponsesWriter) processEvent(event []byte) {
	var eventType string
	var data []byte

	for _, line := range bytes.Split(event, []byte("\n")) {
		line = bytes.TrimRight(line, "\r")
		if bytes.HasPrefix(line, []byte("event:")) {
			eventType = string(bytes.TrimSpace(line[6:]))
		} else if bytes.HasPrefix(line, []byte("event: ")) {
			eventType = string(bytes.TrimSpace(line[7:]))
		} else if bytes.HasPrefix(line, []byte("data: ")) {
			data = bytes.TrimSpace(line[6:])
		} else if bytes.HasPrefix(line, []byte("data:")) {
			data = bytes.TrimSpace(line[5:])
		}
	}

	if len(data) > 0 && cw.callback != nil && !bytes.Equal(data, []byte("[DONE]")) {
		cw.callback(data)
	}

	if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
		return
	}

	var obj map[string]any
	if err := json.Unmarshal(data, &obj); err != nil {
		return
	}

	if eventType == "" {
		if t, ok := obj["type"].(string); ok {
			eventType = t
		}
	}

	switch eventType {
	case "message_start":
		msg, _ := obj["message"].(map[string]any)
		if msg != nil {
			if id, ok := msg["id"].(string); ok {
				cw.id = id
			}
			if model, ok := msg["model"].(string); ok {
				cw.model = model
			}
			if usage, ok := msg["usage"].(map[string]any); ok {
				cw.inputTokens = toInt64(usage["input_tokens"])
			}
		}
		cw.started = true
		// Emit response.created
		cw.emitEvent("response.created", map[string]any{
			"type": "response.created",
			"response": map[string]any{
				"id":     cw.id,
				"object": "response",
				"model":  cw.model,
				"status": "in_progress",
				"output": []any{},
				"usage":  map[string]any{"input_tokens": cw.inputTokens, "output_tokens": 0},
			},
		})

	case "content_block_start":
		block, _ := obj["content_block"].(map[string]any)
		if block == nil {
			return
		}
		blockType, _ := block["type"].(string)
		cw.currentBlockType = blockType

		switch blockType {
		case "text":
			cw.accumulatedText = ""
			// Emit output_item.added for the message (only once)
			if !cw.sentMessageItem {
				cw.sentMessageItem = true
				cw.emitEvent("response.output_item.added", map[string]any{
					"type":         "response.output_item.added",
					"output_index": cw.outputIndex,
					"item": map[string]any{
						"type":    "message",
						"role":    "assistant",
						"content": []any{},
						"status":  "in_progress",
					},
				})
			}
			// Emit content_part.added
			cw.emitEvent("response.content_part.added", map[string]any{
				"type":          "response.content_part.added",
				"output_index":  cw.outputIndex,
				"content_index": cw.contentIndex,
				"part":          map[string]any{"type": "output_text", "text": ""},
			})
		case "thinking":
			// Track thinking block, emit reasoning events
		case "tool_use":
			id, _ := block["id"].(string)
			name, _ := block["name"].(string)
			name = strings.TrimPrefix(name, "mcp_")
			cw.currentToolID = id
			cw.currentToolName = name
			cw.accumulatedArgs = ""
			// Close previous message output item if needed
			if cw.sentMessageItem {
				// Emit content_part.done and output_item.done for the message
				cw.emitEvent("response.content_part.done", map[string]any{
					"type":          "response.content_part.done",
					"output_index":  cw.outputIndex,
					"content_index": cw.contentIndex,
					"part":          map[string]any{"type": "output_text", "text": cw.accumulatedText},
				})
				cw.emitEvent("response.output_item.done", map[string]any{
					"type":         "response.output_item.done",
					"output_index": cw.outputIndex,
					"item": map[string]any{
						"type": "message",
						"role": "assistant",
						"content": []any{
							map[string]any{"type": "output_text", "text": cw.accumulatedText},
						},
						"status": "completed",
					},
				})
				cw.outputIndex++
				cw.contentIndex = 0
				cw.sentMessageItem = false
			}
			// Emit output_item.added for the function_call
			cw.emitEvent("response.output_item.added", map[string]any{
				"type":         "response.output_item.added",
				"output_index": cw.outputIndex,
				"item": map[string]any{
					"type":      "function_call",
					"call_id":   id,
					"name":      name,
					"arguments": "",
					"status":    "in_progress",
				},
			})
		}

	case "content_block_delta":
		delta, _ := obj["delta"].(map[string]any)
		if delta == nil {
			return
		}
		deltaType, _ := delta["type"].(string)
		switch deltaType {
		case "text_delta":
			text, _ := delta["text"].(string)
			if text != "" {
				cw.accumulatedText += text
				cw.emitEvent("response.output_text.delta", map[string]any{
					"type":          "response.output_text.delta",
					"output_index":  cw.outputIndex,
					"content_index": cw.contentIndex,
					"delta":         text,
				})
			}
		case "thinking_delta":
			thinking, _ := delta["thinking"].(string)
			if thinking != "" {
				cw.emitEvent("response.reasoning_text.delta", map[string]any{
					"type":          "response.reasoning_text.delta",
					"output_index":  cw.outputIndex,
					"content_index": cw.contentIndex,
					"delta":         thinking,
				})
			}
		case "input_json_delta":
			partial, _ := delta["partial_json"].(string)
			if partial != "" {
				cw.accumulatedArgs += partial
				cw.emitEvent("response.function_call_arguments.delta", map[string]any{
					"type":         "response.function_call_arguments.delta",
					"output_index": cw.outputIndex,
					"delta":        partial,
				})
			}
		}

	case "content_block_stop":
		switch cw.currentBlockType {
		case "text":
			cw.emitEvent("response.output_text.done", map[string]any{
				"type":          "response.output_text.done",
				"output_index":  cw.outputIndex,
				"content_index": cw.contentIndex,
				"text":          cw.accumulatedText,
			})
			cw.contentIndex++
		case "thinking":
			// No specific done event needed
		case "tool_use":
			cw.emitEvent("response.function_call_arguments.done", map[string]any{
				"type":         "response.function_call_arguments.done",
				"output_index": cw.outputIndex,
				"call_id":      cw.currentToolID,
				"name":         cw.currentToolName,
				"arguments":    cw.accumulatedArgs,
			})
			cw.emitEvent("response.output_item.done", map[string]any{
				"type":         "response.output_item.done",
				"output_index": cw.outputIndex,
				"item": map[string]any{
					"type":      "function_call",
					"call_id":   cw.currentToolID,
					"name":      cw.currentToolName,
					"arguments": cw.accumulatedArgs,
					"status":    "completed",
				},
			})
			cw.outputIndex++
			cw.contentIndex = 0
		}
		cw.currentBlockType = ""

	case "message_delta":
		delta, _ := obj["delta"].(map[string]any)
		if delta != nil {
			if sr, ok := delta["stop_reason"].(string); ok {
				cw.stopReason = sr
			}
		}
		if usage, ok := obj["usage"].(map[string]any); ok {
			cw.outputTokens = toInt64(usage["output_tokens"])
		}

	case "message_stop":
		// Close any open message item
		if cw.sentMessageItem {
			cw.emitEvent("response.content_part.done", map[string]any{
				"type":          "response.content_part.done",
				"output_index":  cw.outputIndex,
				"content_index": cw.contentIndex,
				"part":          map[string]any{"type": "output_text", "text": cw.accumulatedText},
			})
			cw.emitEvent("response.output_item.done", map[string]any{
				"type":         "response.output_item.done",
				"output_index": cw.outputIndex,
				"item": map[string]any{
					"type": "message",
					"role": "assistant",
					"content": []any{
						map[string]any{"type": "output_text", "text": cw.accumulatedText},
					},
					"status": "completed",
				},
			})
		}
		// Emit response.completed
		status := "completed"
		if cw.stopReason == "max_tokens" {
			status = "incomplete"
		}
		cw.emitEvent("response.completed", map[string]any{
			"type": "response.completed",
			"response": map[string]any{
				"id":     cw.id,
				"object": "response",
				"model":  cw.model,
				"status": status,
				"usage": map[string]any{
					"input_tokens":  cw.inputTokens,
					"output_tokens": cw.outputTokens,
					"total_tokens":  cw.inputTokens + cw.outputTokens,
				},
			},
		})

	case "ping":
		// Ignore
	}
}

func (cw *claudeToResponsesWriter) emitEvent(eventType string, data map[string]any) {
	b, err := json.Marshal(data)
	if err != nil {
		return
	}
	out := fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, string(b))
	if _, err := cw.w.Write([]byte(out)); err != nil {
		if cw.debug {
			log.Printf("[%s] claude->responses write error: %v", cw.reqID, err)
		}
	}
}

// responsesToClaudeWriter translates Responses API SSE events to Claude Messages
// API SSE events. Used when Claude Code sends /v1/messages with a Codex model,
// so the Responses API SSE from upstream needs to be converted back to Claude SSE.
type responsesToClaudeWriter struct {
	w        io.Writer
	buf      []byte
	callback func([]byte)
	debug    bool
	reqID    string

	// State
	id                string
	model             string
	started           bool
	contentBlockIndex int
	toolCallIndex     int
	sentText          bool   // whether we've emitted a text content_block_start
	sentThinking      bool   // whether we've emitted a thinking content_block_start
	finishReason      string
	inputTokens       int64
	outputTokens      int64
}

func (rw *responsesToClaudeWriter) Write(p []byte) (int, error) {
	origLen := len(p)
	rw.buf = append(rw.buf, p...)
	rw.scanAndTranslate()
	return origLen, nil
}

func (rw *responsesToClaudeWriter) scanAndTranslate() {
	for {
		idx := bytes.Index(rw.buf, []byte("\n\n"))
		advance := 2
		if idx < 0 {
			idx = bytes.Index(rw.buf, []byte("\r\n\r\n"))
			advance = 4
			if idx < 0 {
				if len(rw.buf) > 1024*1024 {
					rw.buf = rw.buf[len(rw.buf)-512*1024:]
				}
				return
			}
		}
		event := rw.buf[:idx]
		rw.buf = rw.buf[idx+advance:]
		rw.processEvent(event)
	}
}

func (rw *responsesToClaudeWriter) processEvent(event []byte) {
	var eventType string
	var data []byte

	for _, line := range bytes.Split(event, []byte("\n")) {
		line = bytes.TrimRight(line, "\r")
		if bytes.HasPrefix(line, []byte("event:")) {
			eventType = string(bytes.TrimSpace(line[6:]))
		} else if bytes.HasPrefix(line, []byte("event: ")) {
			eventType = string(bytes.TrimSpace(line[7:]))
		} else if bytes.HasPrefix(line, []byte("data: ")) {
			data = bytes.TrimSpace(line[6:])
		} else if bytes.HasPrefix(line, []byte("data:")) {
			data = bytes.TrimSpace(line[5:])
		}
	}

	// Forward original data to usage callback
	if len(data) > 0 && rw.callback != nil && !bytes.Equal(data, []byte("[DONE]")) {
		rw.callback(data)
	}

	if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
		return
	}

	var obj map[string]any
	if err := json.Unmarshal(data, &obj); err != nil {
		if rw.debug {
			log.Printf("[%s] responses->claude: JSON parse error for event %q: %v (data len=%d)", rw.reqID, eventType, err, len(data))
		}
		return
	}

	if eventType == "" {
		if t, ok := obj["type"].(string); ok {
			eventType = t
		}
	}

	if rw.debug {
		log.Printf("[%s] responses->claude: processing event: %s", rw.reqID, eventType)
	}

	switch eventType {
	case "response.created":
		resp, _ := obj["response"].(map[string]any)
		if resp != nil {
			if id, ok := resp["id"].(string); ok {
				rw.id = id
			}
			if m, ok := resp["model"].(string); ok {
				rw.model = m
			}
			if usage, ok := resp["usage"].(map[string]any); ok {
				rw.inputTokens = toInt64(usage["input_tokens"])
			}
		}
		rw.started = true
		// Emit message_start
		rw.emitClaudeMessageStart()

	case "response.output_text.delta":
		delta, _ := obj["delta"].(string)
		if delta != "" {
			// Fallback: if response.created was dropped (e.g. huge event
			// exceeding buffer), emit message_start now.
			if !rw.started {
				rw.started = true
				rw.emitClaudeMessageStart()
			}
			if !rw.sentText {
				rw.sentText = true
				// Close thinking block if it was open
				if rw.sentThinking {
					rw.emitClaudeEvent("content_block_stop", fmt.Sprintf(
						`{"type":"content_block_stop","index":%d}`, rw.contentBlockIndex))
					rw.contentBlockIndex++
				}
				rw.emitClaudeEvent("content_block_start", fmt.Sprintf(
					`{"type":"content_block_start","index":%d,"content_block":{"type":"text","text":""}}`,
					rw.contentBlockIndex))
			}
			rw.emitClaudeEvent("content_block_delta", fmt.Sprintf(
				`{"type":"content_block_delta","index":%d,"delta":{"type":"text_delta","text":%s}}`,
				rw.contentBlockIndex, mustMarshalString(delta)))
		}

	case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
		delta, _ := obj["delta"].(string)
		if delta != "" {
			if !rw.started {
				rw.started = true
				rw.emitClaudeMessageStart()
			}
			if !rw.sentThinking {
				rw.sentThinking = true
				rw.emitClaudeEvent("content_block_start", fmt.Sprintf(
					`{"type":"content_block_start","index":%d,"content_block":{"type":"thinking","thinking":""}}`,
					rw.contentBlockIndex))
			}
			rw.emitClaudeEvent("content_block_delta", fmt.Sprintf(
				`{"type":"content_block_delta","index":%d,"delta":{"type":"thinking_delta","thinking":%s}}`,
				rw.contentBlockIndex, mustMarshalString(delta)))
		}

	case "response.output_item.added":
		item, _ := obj["item"].(map[string]any)
		if item == nil {
			return
		}
		itemType, _ := item["type"].(string)
		if itemType == "function_call" {
			if !rw.started {
				rw.started = true
				rw.emitClaudeMessageStart()
			}
			// Close previous content block
			if rw.sentText || rw.sentThinking {
				rw.emitClaudeEvent("content_block_stop", fmt.Sprintf(
					`{"type":"content_block_stop","index":%d}`, rw.contentBlockIndex))
				rw.contentBlockIndex++
				rw.sentText = false
				rw.sentThinking = false
			}
			callID, _ := item["call_id"].(string)
			name, _ := item["name"].(string)
			rw.emitClaudeEvent("content_block_start", fmt.Sprintf(
				`{"type":"content_block_start","index":%d,"content_block":{"type":"tool_use","id":%s,"name":%s,"input":{}}}`,
				rw.contentBlockIndex, mustMarshalString(callID), mustMarshalString(name)))
			rw.toolCallIndex++
		}

	case "response.function_call_arguments.delta":
		delta, _ := obj["delta"].(string)
		if delta != "" {
			rw.emitClaudeEvent("content_block_delta", fmt.Sprintf(
				`{"type":"content_block_delta","index":%d,"delta":{"type":"input_json_delta","partial_json":%s}}`,
				rw.contentBlockIndex, mustMarshalString(delta)))
		}

	case "response.output_item.done":
		item, _ := obj["item"].(map[string]any)
		if item != nil {
			if itemType, _ := item["type"].(string); itemType == "function_call" {
				rw.emitClaudeEvent("content_block_stop", fmt.Sprintf(
					`{"type":"content_block_stop","index":%d}`, rw.contentBlockIndex))
				rw.contentBlockIndex++
			}
		}

	case "response.completed":
		resp, _ := obj["response"].(map[string]any)
		if resp != nil {
			if usage, ok := resp["usage"].(map[string]any); ok {
				rw.inputTokens = toInt64(usage["input_tokens"])
				rw.outputTokens = toInt64(usage["output_tokens"])
			}
		}
		// Close any open content block
		if rw.sentText || rw.sentThinking {
			rw.emitClaudeEvent("content_block_stop", fmt.Sprintf(
				`{"type":"content_block_stop","index":%d}`, rw.contentBlockIndex))
		}
		// Determine stop reason
		stopReason := "end_turn"
		if resp != nil {
			if status, ok := resp["status"].(string); ok && status == "incomplete" {
				stopReason = "max_tokens"
			}
		}
		if rw.toolCallIndex > 0 {
			stopReason = "tool_use"
		}
		rw.finishReason = stopReason
		// Emit message_delta and message_stop
		rw.emitClaudeEvent("message_delta", fmt.Sprintf(
			`{"type":"message_delta","delta":{"stop_reason":%s,"stop_sequence":null},"usage":{"output_tokens":%d}}`,
			mustMarshalString(stopReason), rw.outputTokens))
		rw.emitClaudeEvent("message_stop", `{"type":"message_stop"}`)

	case "response.failed":
		resp, _ := obj["response"].(map[string]any)
		errMsg := "response failed"
		if resp != nil {
			if e, ok := resp["error"].(map[string]any); ok {
				if m, ok := e["message"].(string); ok && m != "" {
					errMsg = m
				}
			}
		}
		// Emit error as text content
		if !rw.sentText {
			rw.sentText = true
			rw.emitClaudeEvent("content_block_start", fmt.Sprintf(
				`{"type":"content_block_start","index":%d,"content_block":{"type":"text","text":""}}`,
				rw.contentBlockIndex))
		}
		rw.emitClaudeEvent("content_block_delta", fmt.Sprintf(
			`{"type":"content_block_delta","index":%d,"delta":{"type":"text_delta","text":%s}}`,
			rw.contentBlockIndex, mustMarshalString("[Error: "+errMsg+"]")))
		rw.emitClaudeEvent("content_block_stop", fmt.Sprintf(
			`{"type":"content_block_stop","index":%d}`, rw.contentBlockIndex))
		rw.emitClaudeEvent("message_delta", fmt.Sprintf(
			`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":0}}`))
		rw.emitClaudeEvent("message_stop", `{"type":"message_stop"}`)

	case "response.output_text.done", "response.content_part.done",
		"response.content_part.added", "response.reasoning_text.done",
		"response.reasoning_summary_text.done", "response.function_call_arguments.done",
		"response.in_progress":
		// Informational events, no action needed

	default:
		if rw.debug {
			log.Printf("[%s] responses->claude: unhandled event type: %s", rw.reqID, eventType)
		}
	}
}

func (rw *responsesToClaudeWriter) emitClaudeMessageStart() {
	model := rw.model
	if model == "" {
		model = "unknown"
	}
	id := rw.id
	if id == "" {
		id = "msg_translated"
	}
	msg := fmt.Sprintf(`{"type":"message_start","message":{"id":%s,"type":"message","role":"assistant","model":%s,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":%d,"output_tokens":0}}}`,
		mustMarshalString(id), mustMarshalString(model), rw.inputTokens)
	rw.emitClaudeEvent("message_start", msg)
}

func (rw *responsesToClaudeWriter) emitClaudeEvent(eventType, data string) {
	out := fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, data)
	if rw.debug {
		preview := data
		if len(preview) > 200 {
			preview = preview[:200] + "..."
		}
		log.Printf("[%s] responses->claude EMIT: %s (len=%d) %s", rw.reqID, eventType, len(data), preview)
	}
	if _, err := rw.w.Write([]byte(out)); err != nil {
		if rw.debug {
			log.Printf("[%s] responses->claude write error: %v", rw.reqID, err)
		}
	}
}
