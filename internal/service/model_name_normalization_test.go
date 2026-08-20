package service

import "testing"

func TestNormalizeListedModelName(t *testing.T) {
	remote := []string{
		"BAAI/bge-small-en-v1.5",
		"claude-sonnet-5",
		"model",
		"text-embedding-nomic-embed-text-v1.5@q8_0",
	}

	tests := []struct {
		name      string
		requested string
		want      string
		changed   bool
	}{
		{name: "exact remote id", requested: "claude-sonnet-5", want: "claude-sonnet-5"},
		{name: "legacy routing suffix", requested: "claude-sonnet-5@openai", want: "claude-sonnet-5", changed: true},
		{name: "embedding routing suffix", requested: "BAAI/bge-small-en-v1.5@openai", want: "BAAI/bge-small-en-v1.5", changed: true},
		{name: "legitimate at sign", requested: "text-embedding-nomic-embed-text-v1.5@q8_0", want: "text-embedding-nomic-embed-text-v1.5@q8_0"},
		{name: "unproven suffix is preserved", requested: "unknown@openai", want: "unknown@openai"},
		{name: "version suffix is preserved", requested: "model@v2", want: "model@v2"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, changed := normalizeListedModelName(tt.requested, remote)
			if got != tt.want || changed != tt.changed {
				t.Fatalf("normalizeListedModelName(%q) = (%q, %v), want (%q, %v)", tt.requested, got, changed, tt.want, tt.changed)
			}
		})
	}
}
