package dataset

import (
	"math"
	"reflect"
	"testing"
)

func TestDatasetChunkIDUsesAvailableElasticsearchID(t *testing.T) {
	tests := []struct {
		name  string
		chunk map[string]interface{}
		want  string
	}{
		{
			name:  "canonical chunk id has priority",
			chunk: map[string]interface{}{"chunk_id": "canonical", "id": "public", "_id": "internal"},
			want:  "canonical",
		},
		{
			name:  "public elasticsearch id",
			chunk: map[string]interface{}{"id": "public", "_id": "internal"},
			want:  "public",
		},
		{
			name:  "internal elasticsearch id fallback",
			chunk: map[string]interface{}{"_id": "internal"},
			want:  "internal",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := datasetChunkID(tt.chunk); got != tt.want {
				t.Fatalf("datasetChunkID() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDatasetGuessVecFieldAcceptsJSONDecodedVector(t *testing.T) {
	src := map[string]interface{}{
		"content": "sample",
		"q_3_vec": []interface{}{0.25, -0.5, 1.0},
	}

	if got := datasetGuessVecField(src); got != "q_3_vec" {
		t.Fatalf("datasetGuessVecField() = %q, want q_3_vec", got)
	}
}

func TestDatasetGuessVecFieldRejectsDimensionMismatch(t *testing.T) {
	src := map[string]interface{}{
		"q_384_vec": []interface{}{0.25, -0.5, 1.0},
	}

	if got := datasetGuessVecField(src); got != "" {
		t.Fatalf("datasetGuessVecField() = %q, want empty", got)
	}
}

func TestDatasetGuessVecFieldRejectsZeroDimension(t *testing.T) {
	src := map[string]interface{}{
		"q_0_vec": []interface{}{},
	}

	if got := datasetGuessVecField(src); got != "" {
		t.Fatalf("datasetGuessVecField() = %q, want empty", got)
	}
}

func TestDatasetGuessVecFieldRejectsInvalidVectors(t *testing.T) {
	tests := []struct {
		name string
		vec  interface{}
	}{
		{name: "empty", vec: []interface{}{}},
		{name: "nonnumeric", vec: []interface{}{0.25, "invalid", 1.0}},
		{name: "nonfinite", vec: []interface{}{0.25, math.Inf(1)}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := datasetGuessVecField(map[string]interface{}{"q_384_vec": tt.vec}); got != "" {
				t.Fatalf("datasetGuessVecField() = %q, want empty", got)
			}
		})
	}
}

func TestDatasetGuessVecFieldSelectsDeterministically(t *testing.T) {
	src := map[string]interface{}{
		"u_1_vec": []interface{}{1.0},
		"q_1_vec": []interface{}{1.0},
	}

	for range 100 {
		if got := datasetGuessVecField(src); got != "q_1_vec" {
			t.Fatalf("datasetGuessVecField() = %q, want q_1_vec", got)
		}
	}
}

func TestDatasetGuessVecFieldRejectsAmbiguousQuestionVectors(t *testing.T) {
	src := map[string]interface{}{
		"q_1_vec": []interface{}{1.0},
		"q_2_vec": []interface{}{1.0, 0.0},
	}

	if got := datasetGuessVecField(src); got != "" {
		t.Fatalf("datasetGuessVecField() = %q, want empty for ambiguous vectors", got)
	}
}

func TestDatasetGuessVecFieldRejectsMalformedVectorField(t *testing.T) {
	src := map[string]interface{}{
		"q_custom_vec": []interface{}{1.0},
	}

	if got := datasetGuessVecField(src); got != "" {
		t.Fatalf("datasetGuessVecField() = %q, want empty", got)
	}
}

func TestDatasetAsFloatVecAcceptsEngineRepresentations(t *testing.T) {
	tests := []struct {
		name string
		vec  interface{}
		want []float64
	}{
		{name: "elasticsearch tab-separated string", vec: "0.25\t-0.5\t1", want: []float64{0.25, -0.5, 1}},
		{name: "infinity float32 slice", vec: []float32{0.25, -0.5, 1}, want: []float64{0.25, -0.5, 1}},
		{name: "mixed decoded values", vec: []interface{}{float32(0.25), "-0.5", 1.0}, want: []float64{0.25, -0.5, 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := datasetAsFloatVec(tt.vec); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("datasetAsFloatVec() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestDatasetAsFloatVecRejectsInvalidRepresentations(t *testing.T) {
	tests := []struct {
		name string
		vec  interface{}
	}{
		{name: "partial decoded vector", vec: []interface{}{0.25, "invalid", 1.0}},
		{name: "invalid tab-separated string", vec: "0.25\tinvalid\t1"},
		{name: "nonfinite tab-separated string", vec: "0.25\tNaN\t1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := datasetAsFloatVec(tt.vec); got != nil {
				t.Fatalf("datasetAsFloatVec() = %#v, want nil", got)
			}
		})
	}
}
