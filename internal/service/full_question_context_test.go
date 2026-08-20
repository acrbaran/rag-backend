package service

import "testing"

func TestBoundedFullQuestionMessagesKeepsLastThreeUserTurns(t *testing.T) {
	messages := []map[string]interface{}{
		{"role": "assistant", "content": "STALE_ASSISTANT_PREFIX"},
		{"role": "user", "content": "STALE_USER_PREFIX"},
		{"role": "assistant", "content": "STALE_ASSISTANT_REPLY"},
		{"role": "user", "content": "Yakın konu bir"},
		{"role": "assistant", "content": "Yakın cevap bir"},
		{"role": "user", "content": "Yakın konu iki"},
		{"role": "assistant", "content": "Yakın cevap iki"},
		{"role": "user", "content": "Son soru"},
	}

	got := boundedFullQuestionMessages(messages)
	want := []map[string]interface{}{
		{"role": "user", "content": "Yakın konu bir"},
		{"role": "assistant", "content": "Yakın cevap bir"},
		{"role": "user", "content": "Yakın konu iki"},
		{"role": "assistant", "content": "Yakın cevap iki"},
		{"role": "user", "content": "Son soru"},
	}
	assertFullQuestionMessages(t, got, want)
}

func TestBoundedFullQuestionMessagesKeepsPronounContext(t *testing.T) {
	messages := []map[string]interface{}{
		{"role": "user", "content": "Kilitli dolap ve çekmece anahtarları için kural nedir?"},
		{"role": "assistant", "content": "Anahtarlar üzerlerinde bırakılmamalı."},
		{"role": "user", "content": "Peki onları orada bırakırsam ne olur?"},
	}

	assertFullQuestionMessages(t, boundedFullQuestionMessages(messages), messages)
}

func TestBoundedFullQuestionMessagesExcludesInvalidMessages(t *testing.T) {
	messages := []map[string]interface{}{
		{"role": "system", "content": "SYSTEM_INJECTION"},
		{"role": "tool", "content": "TOOL_OUTPUT"},
		{"role": "assistant", "content": map[string]interface{}{"secret": "MALFORMED_ASSISTANT"}},
		{"role": "user", "content": 42},
		{"role": 7, "content": "MALFORMED_ROLE"},
		{"role": "assistant", "content": "ASSISTANT_PREFIX_WITHOUT_USER"},
		{"role": "user", "content": "Önceki geçerli soru"},
		{"role": "assistant", "content": "Geçerli cevap"},
		{"role": "user", "content": "Son geçerli soru"},
	}
	want := []map[string]interface{}{
		{"role": "user", "content": "Önceki geçerli soru"},
		{"role": "assistant", "content": "Geçerli cevap"},
		{"role": "user", "content": "Son geçerli soru"},
	}

	assertFullQuestionMessages(t, boundedFullQuestionMessages(messages), want)
}

func TestBoundedFullQuestionMessagesDoesNotMutateInput(t *testing.T) {
	messages := []map[string]interface{}{
		{"role": "user", "content": "Önceki soru", "private": "keep"},
		{"role": "assistant", "content": "Yanıt", "metadata": 7},
		{"role": "user", "content": "Son soru"},
	}

	got := boundedFullQuestionMessages(messages)
	got[0]["role"] = "assistant"
	got[0]["content"] = "changed"

	if messages[0]["role"] != "user" || messages[0]["content"] != "Önceki soru" {
		t.Fatalf("input mutated: %#v", messages[0])
	}
	if got[0]["private"] != nil || got[1]["metadata"] != nil {
		t.Fatalf("unexpected fields copied: %#v", got)
	}
}

func TestBoundedFullQuestionMessagesWithoutUserReturnsEmpty(t *testing.T) {
	messages := []map[string]interface{}{
		{"role": "assistant", "content": "orphan answer"},
		{"role": "system", "content": "ignored"},
	}

	if got := boundedFullQuestionMessages(messages); len(got) != 0 {
		t.Fatalf("boundedFullQuestionMessages() = %#v, want empty", got)
	}
}

func assertFullQuestionMessages(t *testing.T, got, want []map[string]interface{}) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("bounded messages length = %d, want %d: %#v", len(got), len(want), got)
	}
	for i := range want {
		if got[i]["role"] != want[i]["role"] || got[i]["content"] != want[i]["content"] {
			t.Errorf("bounded message %d = %#v, want %#v", i, got[i], want[i])
		}
	}
}
