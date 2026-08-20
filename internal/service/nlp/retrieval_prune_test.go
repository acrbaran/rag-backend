package nlp

import (
	"reflect"
	"testing"
)

func TestPruneRetrievalSearchResultPreservesRawEngineIdentity(t *testing.T) {
	liveRaw := map[string]interface{}{
		"_id":                 "live-chunk",
		"id":                  "live-chunk",
		"_index":              "ragflow_tenant",
		"doc_id":              "live-doc",
		"content_with_weight": "live content",
	}
	staleRaw := map[string]interface{}{
		"_id":    "stale-chunk",
		"id":     "stale-chunk",
		"_index": "ragflow_tenant",
		"doc_id": "stale-doc",
	}
	result := &RetrievalSearchResult{
		Chunks:      []map[string]interface{}{staleRaw, liveRaw},
		Total:       2,
		QueryVector: []float64{0.1, 0.2},
		Highlight:   map[string]string{"stale-chunk": "stale", "live-chunk": "live"},
		Field: map[string]map[string]interface{}{
			"stale-chunk": {"doc_id": "stale-doc", "content_with_weight": "stale content"},
			"live-chunk":  {"doc_id": "live-doc", "content_with_weight": "live content"},
		},
		IDs:         []string{"stale-chunk", "live-chunk"},
		Keywords:    []string{"policy"},
		Aggregation: []map[string]interface{}{{"doc_name": "policy.pdf"}},
		Options:     map[string]interface{}{"total": 2},
		IndexNames:  []string{"ragflow_tenant"},
	}

	filtered, removed := pruneRetrievalSearchResult(result, map[string]struct{}{"live-doc": {}})

	if removed != 1 {
		t.Fatalf("removed=%d, want 1", removed)
	}
	if filtered.Total != 1 || !reflect.DeepEqual(filtered.IDs, []string{"live-chunk"}) {
		t.Fatalf("filtered result total=%d ids=%v", filtered.Total, filtered.IDs)
	}
	if len(filtered.Chunks) != 1 || filtered.Chunks[0]["_id"] != "live-chunk" || filtered.Chunks[0]["_index"] != "ragflow_tenant" {
		t.Fatalf("raw engine identity was not preserved: %#v", filtered.Chunks)
	}
	if !reflect.DeepEqual(filtered.IndexNames, result.IndexNames) {
		t.Fatalf("index names=%v, want %v", filtered.IndexNames, result.IndexNames)
	}
	if !reflect.DeepEqual(filtered.Highlight, map[string]string{"live-chunk": "live"}) {
		t.Fatalf("highlights=%v", filtered.Highlight)
	}
}
