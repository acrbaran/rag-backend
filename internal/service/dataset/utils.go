package dataset

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"ragflow/internal/entity"
	modelModule "ragflow/internal/entity/models"
	"ragflow/internal/service"
)

func datasetListItemToMap(kb *entity.KnowledgebaseListItem) map[string]interface{} {
	// avatar/language/description keys are always present (null when unset),
	// matching Python's full-row dict response.
	item := map[string]interface{}{
		"id":              kb.ID,
		"name":            kb.Name,
		"avatar":          stringPointerValue(kb.Avatar),
		"language":        stringPointerValue(kb.Language),
		"description":     stringPointerValue(kb.Description),
		"tenant_id":       kb.TenantID,
		"permission":      kb.Permission,
		"document_count":  kb.DocNum,
		"token_num":       kb.TokenNum,
		"chunk_count":     kb.ChunkNum,
		"parser_id":       kb.ParserID,
		"parser_config":   jsonMapValue(kb.ParserConfig),
		"pagerank":        kb.Pagerank,
		"embedding_model": kb.EmbdID,
		"nickname":        kb.Nickname,
	}
	if kb.TenantAvatar != nil {
		item["tenant_avatar"] = *kb.TenantAvatar
	}
	if kb.UpdateTime != nil {
		item["update_time"] = *kb.UpdateTime
	}
	return item
}

func datasetToMap(kb *entity.Knowledgebase) map[string]interface{} {
	item := map[string]interface{}{
		"id":                       kb.ID,
		"tenant_id":                kb.TenantID,
		"name":                     kb.Name,
		"embedding_model":          kb.EmbdID,
		"permission":               kb.Permission,
		"created_by":               kb.CreatedBy,
		"document_count":           kb.DocNum,
		"token_num":                kb.TokenNum,
		"chunk_count":              kb.ChunkNum,
		"similarity_threshold":     kb.SimilarityThreshold,
		"vector_similarity_weight": kb.VectorSimilarityWeight,
		"parser_id":                kb.ParserID,
		"parser_config":            kb.ParserConfig,
		"pagerank":                 kb.Pagerank,
		"create_time":              kb.CreateTime,
	}
	if kb.Avatar != nil {
		item["avatar"] = *kb.Avatar
	}
	if kb.Language != nil {
		item["language"] = *kb.Language
	}
	if kb.Description != nil {
		item["description"] = *kb.Description
	}
	if kb.PipelineID != nil {
		item["pipeline_id"] = *kb.PipelineID
	}
	if kb.GraphragTaskID != nil {
		item["graphrag_task_id"] = *kb.GraphragTaskID
	}
	if kb.GraphragTaskFinishAt != nil {
		item["graphrag_task_finish_at"] = kb.GraphragTaskFinishAt.Format("2006-01-02 15:04:05")
	}
	if kb.RaptorTaskID != nil {
		item["raptor_task_id"] = *kb.RaptorTaskID
	}
	if kb.RaptorTaskFinishAt != nil {
		item["raptor_task_finish_at"] = kb.RaptorTaskFinishAt.Format("2006-01-02 15:04:05")
	}
	if kb.MindmapTaskID != nil {
		item["mindmap_task_id"] = *kb.MindmapTaskID
	}
	if kb.MindmapTaskFinishAt != nil {
		item["mindmap_task_finish_at"] = kb.MindmapTaskFinishAt.Format("2006-01-02 15:04:05")
	}
	if kb.UpdateTime != nil {
		item["update_time"] = *kb.UpdateTime
	}
	return item
}

func limitStrings(values []string, limit int) []string {
	if len(values) <= limit {
		return values
	}
	return values[:limit]
}

func stringPointerValue(s *string) interface{} {
	if s == nil {
		return nil
	}
	return *s
}

func int64PointerValue(i *int64) interface{} {
	if i == nil {
		return nil
	}
	return *i
}

func timePointerValue(t *time.Time) interface{} {
	if t == nil {
		return nil
	}
	return t.Format("2006-01-02 15:04:05")
}

func jsonMapValue(m entity.JSONMap) interface{} {
	if m == nil {
		return nil
	}
	return map[string]interface{}(m)
}

func datasetMap(value interface{}) map[string]interface{} {
	if m, ok := value.(map[string]interface{}); ok {
		return m
	}
	return nil
}

func datasetString(value interface{}) string {
	if s, ok := value.(string); ok {
		return s
	}
	return ""
}

func datasetStringSlice(value interface{}) []string {
	if sl, ok := value.([]string); ok {
		return sl
	}
	if raw, ok := value.([]interface{}); ok {
		result := make([]string, 0, len(raw))
		for _, v := range raw {
			if s, ok := v.(string); ok {
				result = append(result, s)
			}
		}
		return result
	}
	return nil
}

var datasetVectorFieldPattern = regexp.MustCompile(`^[qu]_(\d+)_vec$`)

func datasetGuessVecField(src map[string]interface{}) string {
	questionFields := make([]string, 0, 1)
	fallbackFields := make([]string, 0, 1)
	for k, v := range src {
		matches := datasetVectorFieldPattern.FindStringSubmatch(k)
		if len(matches) != 2 {
			continue
		}
		dimension, err := strconv.Atoi(matches[1])
		vector := datasetAsFloatVec(v)
		if err != nil || dimension <= 0 || len(vector) == 0 || dimension != len(vector) {
			continue
		}
		if strings.HasPrefix(k, "q_") {
			questionFields = append(questionFields, k)
		} else {
			fallbackFields = append(fallbackFields, k)
		}
	}
	if len(questionFields) == 1 {
		return questionFields[0]
	}
	if len(questionFields) > 1 || len(fallbackFields) != 1 {
		return ""
	}
	return fallbackFields[0]
}

func datasetAsFloatVec(v interface{}) []float64 {
	var vec []float64
	switch val := v.(type) {
	case string:
		parts := strings.Split(val, "\t")
		vec = make([]float64, len(parts))
		for i, part := range parts {
			n, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
			if err != nil {
				return nil
			}
			vec[i] = n
		}
	case []float64:
		vec = append([]float64(nil), val...)
	case []float32:
		vec = make([]float64, len(val))
		for i, n := range val {
			vec[i] = float64(n)
		}
	case []interface{}:
		vec = make([]float64, len(val))
		for i, item := range val {
			switch n := item.(type) {
			case float64:
				vec[i] = n
			case float32:
				vec[i] = float64(n)
			case int:
				vec[i] = float64(n)
			case int64:
				vec[i] = float64(n)
			case json.Number:
				f, err := n.Float64()
				if err != nil {
					return nil
				}
				vec[i] = f
			case string:
				f, err := strconv.ParseFloat(strings.TrimSpace(n), 64)
				if err != nil {
					return nil
				}
				vec[i] = f
			default:
				return nil
			}
		}
	default:
		return nil
	}
	if len(vec) == 0 {
		return nil
	}
	for _, n := range vec {
		if math.IsNaN(n) || math.IsInf(n, 0) {
			return nil
		}
	}
	return vec
}

func datasetCosSim(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func datasetCleanEmbeddingText(s string) string {
	re := regexp.MustCompile(`<[^>]*>`)
	return re.ReplaceAllString(s, "")
}

func datasetEncodeEmbedding(ctx context.Context, embeddingModel *modelModule.EmbeddingModel, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	cleaned := make([]string, len(texts))
	for i, t := range texts {
		cleaned[i] = datasetCleanEmbeddingText(t)
	}
	embeddingConfig := &modelModule.EmbeddingConfig{Dimension: 0}
	embeddings, err := embeddingModel.ModelDriver.Embed(ctx, embeddingModel.ModelName, modelModule.EmbedRequest{Texts: cleaned}, embeddingModel.APIConfig, embeddingConfig, nil)
	if err != nil {
		return nil, err
	}
	if len(embeddings) != len(cleaned) {
		return nil, fmt.Errorf(
			"embedding response count %d does not match input count %d",
			len(embeddings),
			len(cleaned),
		)
	}
	vectors := make([][]float64, len(embeddings))
	for i, embedding := range embeddings {
		if len(embedding.Embedding) == 0 {
			return nil, fmt.Errorf("empty embedding vector at index %d", i)
		}
		for j, value := range embedding.Embedding {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf(
					"non-finite embedding value at vector %d index %d",
					i,
					j,
				)
			}
		}
		vectors[i] = embedding.Embedding
	}
	return vectors, nil
}

func datasetMixVectors(titleVector, contentVector []float64, titleWeight float64) []float64 {
	if len(titleVector) == 0 && len(contentVector) == 0 {
		return nil
	}
	if len(titleVector) == 0 {
		return contentVector
	}
	if len(contentVector) == 0 {
		return titleVector
	}
	minLen := len(titleVector)
	if len(contentVector) < minLen {
		minLen = len(contentVector)
	}
	mixed := make([]float64, minLen)
	for i := 0; i < minLen; i++ {
		mixed[i] = titleWeight*titleVector[i] + (1-titleWeight)*contentVector[i]
	}
	return mixed
}

func datasetEmbeddingCheckSummary(datasetID, embeddingID string, sampled int, similarities []float64, matchMode string) service.EmbeddingCheckSummary {
	if len(similarities) == 0 {
		return service.EmbeddingCheckSummary{
			KbID:      datasetID,
			Model:     embeddingID,
			Sampled:   sampled,
			Valid:     0,
			AvgCosSim: 0,
			MinCosSim: 0,
			MaxCosSim: 0,
			MatchMode: matchMode,
		}
	}
	sort.Float64s(similarities)
	var sum float64
	for _, v := range similarities {
		sum += v
	}
	return service.EmbeddingCheckSummary{
		KbID:      datasetID,
		Model:     embeddingID,
		Sampled:   sampled,
		Valid:     len(similarities),
		AvgCosSim: datasetRoundFloat(sum/float64(len(similarities)), 4),
		MinCosSim: datasetRoundFloat(similarities[0], 4),
		MaxCosSim: datasetRoundFloat(similarities[len(similarities)-1], 4),
		MatchMode: matchMode,
	}
}

func datasetRoundFloat(value float64, places int) float64 {
	shift := math.Pow(10, float64(places))
	return math.Round(value*shift) / shift
}

func datasetChunkID(chunk map[string]interface{}) string {
	for _, field := range []string{"chunk_id", "id", "_id"} {
		if id, ok := chunk[field].(string); ok && id != "" {
			return id
		}
	}
	return ""
}

func interfaceSlice(items ...string) []interface{} {
	result := make([]interface{}, len(items))
	for i, v := range items {
		result[i] = v
	}
	return result
}
