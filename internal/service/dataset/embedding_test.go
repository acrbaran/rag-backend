package dataset

import (
	"context"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"

	"ragflow/internal/common"
	"ragflow/internal/dao"
	"ragflow/internal/engine"
	enginetypes "ragflow/internal/engine/types"
	"ragflow/internal/entity"
	modelModule "ragflow/internal/entity/models"
	serviceModule "ragflow/internal/service"
	"ragflow/internal/utility"
)

type embeddingSampleEngine struct {
	engine.DocEngine
	chunks map[string]map[string]interface{}
}

type embeddingReplacementEngine struct {
	embeddingSampleEngine
	searchCalls int
}

type embeddingSingleReadEngine struct {
	embeddingSampleEngine
	getCalls int
}

type embeddingSearchFailureEngine struct {
	engine.DocEngine
	response *enginetypes.SearchResult
	err      error
}

func (e *embeddingSearchFailureEngine) Search(
	context.Context,
	*enginetypes.SearchRequest,
) (*enginetypes.SearchResult, error) {
	return e.response, e.err
}

type embeddingGetChunkFailureEngine struct {
	embeddingSampleEngine
	response interface{}
	err      error
}

type embeddingMalformedPageEngine struct {
	embeddingSampleEngine
	chunk map[string]interface{}
}

type embeddingProtocolDriver struct {
	modelModule.ModelDriver
	vectors [][]float64
	calls   int
}

func (d *embeddingProtocolDriver) Embed(
	context.Context,
	*string,
	modelModule.EmbedRequest,
	*modelModule.APIConfig,
	*modelModule.EmbeddingConfig,
	*common.ModelUsage,
) ([]modelModule.EmbeddingData, error) {
	if d.calls >= len(d.vectors) {
		return nil, errors.New("unexpected embedding call")
	}
	vector := d.vectors[d.calls]
	d.calls++
	return []modelModule.EmbeddingData{{Embedding: vector}}, nil
}

func (e *embeddingMalformedPageEngine) Search(
	_ context.Context,
	req *enginetypes.SearchRequest,
) (*enginetypes.SearchResult, error) {
	if len(req.SelectFields) == 0 {
		return &enginetypes.SearchResult{Total: 1}, nil
	}
	chunks := []map[string]interface{}{}
	if e.chunk != nil {
		chunks = append(chunks, e.chunk)
	}
	return &enginetypes.SearchResult{Total: 1, Chunks: chunks}, nil
}

func (e *embeddingGetChunkFailureEngine) GetChunk(
	context.Context,
	string,
	string,
	[]string,
) (interface{}, error) {
	return e.response, e.err
}

func (e *embeddingSingleReadEngine) GetChunk(ctx context.Context, indexName, chunkID string, kbIDs []string) (interface{}, error) {
	e.getCalls++
	if e.getCalls > 1 {
		return nil, errors.New("unexpected second chunk read")
	}
	return e.embeddingSampleEngine.GetChunk(ctx, indexName, chunkID, kbIDs)
}

func (e *embeddingReplacementEngine) Search(_ context.Context, req *enginetypes.SearchRequest) (*enginetypes.SearchResult, error) {
	if len(req.SelectFields) == 0 {
		return &enginetypes.SearchResult{Total: int64(len(e.chunks))}, nil
	}
	e.searchCalls++
	id := "invalid"
	if e.searchCalls > 1 {
		id = "valid"
	}
	return &enginetypes.SearchResult{
		Total:  int64(len(e.chunks)),
		Chunks: []map[string]interface{}{{"id": id}},
	}, nil
}

func (e *embeddingSampleEngine) Search(_ context.Context, req *enginetypes.SearchRequest) (*enginetypes.SearchResult, error) {
	if len(req.SelectFields) == 0 {
		return &enginetypes.SearchResult{Total: int64(len(e.chunks))}, nil
	}
	ids := make([]string, 0, len(e.chunks))
	for id := range e.chunks {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	if req.Offset < 0 || req.Offset >= len(ids) {
		return &enginetypes.SearchResult{Total: int64(len(ids))}, nil
	}
	id := ids[req.Offset]
	return &enginetypes.SearchResult{
		Total:  int64(len(ids)),
		Chunks: []map[string]interface{}{{"id": id}},
	}, nil
}

func (e *embeddingSampleEngine) GetChunk(_ context.Context, _, chunkID string, _ []string) (interface{}, error) {
	return e.chunks[chunkID], nil
}

func setupEmbeddingCheckService(
	t *testing.T,
	baseURL string,
	docEngine engine.DocEngine,
) *DatasetService {
	t.Helper()
	previousAllowAnyHost := utility.AllowAnyHostForTest
	utility.AllowAnyHostForTest = true
	t.Cleanup(func() { utility.AllowAnyHostForTest = previousAllowAnyHost })

	db := setupServiceTestDB(t)
	pushServiceDB(t, db)

	status := "1"
	rows := []interface{}{
		&entity.Knowledgebase{
			ID:         "dataset-1",
			TenantID:   "user-1",
			Name:       "test dataset",
			EmbdID:     "current-model",
			Permission: string(entity.TenantPermissionMe),
			CreatedBy:  "user-1",
			Status:     &status,
		},
		&entity.TenantModelProvider{
			ID:           "provider-1",
			TenantID:     "user-1",
			ProviderName: "OpenAI",
		},
		&entity.TenantModelInstance{
			ID:           "instance-1",
			ProviderID:   "provider-1",
			InstanceName: "default",
			APIKey:       "test-key",
			Status:       "active",
			Extra:        `{"base_url":"` + baseURL + `"}`,
		},
		&entity.TenantModel{
			ID:         "candidate-model",
			ProviderID: "provider-1",
			InstanceID: "instance-1",
			ModelName:  "tenant-only-embedding-check",
			ModelType:  int(entity.ModelTypeEmbedding),
			Status:     "active",
			Extra:      `{"max_dimension":2,"max_batch_size":1,"dimensions":[2]}`,
		},
	}
	for _, row := range rows {
		if err := db.Create(row).Error; err != nil {
			t.Fatalf("failed to seed %T: %v", row, err)
		}
	}

	service := NewDatasetService()
	service.docEngine = docEngine
	return service
}

func TestCheckEmbeddingUsesSampledChunkWithoutSecondRead(t *testing.T) {
	db := setupServiceTestDB(t)
	pushServiceDB(t, db)

	previousAllowAnyHost := utility.AllowAnyHostForTest
	utility.AllowAnyHostForTest = true
	t.Cleanup(func() { utility.AllowAnyHostForTest = previousAllowAnyHost })

	var requestPath string
	var requestPathMu sync.Mutex
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestPathMu.Lock()
		requestPath = r.URL.Path
		requestPathMu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[1,0],"index":0}]}`))
	}))
	t.Cleanup(server.Close)

	status := "1"
	rows := []interface{}{
		&entity.Knowledgebase{
			ID:         "dataset-1",
			TenantID:   "user-1",
			Name:       "test dataset",
			EmbdID:     "current-model",
			Permission: string(entity.TenantPermissionMe),
			CreatedBy:  "user-1",
			Status:     &status,
		},
		&entity.TenantModelProvider{
			ID:           "provider-1",
			TenantID:     "user-1",
			ProviderName: "OpenAI",
		},
		&entity.TenantModelInstance{
			ID:           "instance-1",
			ProviderID:   "provider-1",
			InstanceName: "default",
			APIKey:       "test-key",
			Status:       "active",
			Extra:        `{"base_url":"` + server.URL + `"}`,
		},
		&entity.TenantModel{
			ID:         "candidate-model",
			ProviderID: "provider-1",
			InstanceID: "instance-1",
			ModelName:  "tenant-only-embedding-check",
			ModelType:  int(entity.ModelTypeEmbedding),
			Status:     "active",
			Extra:      `{"max_dimension":2,"max_batch_size":1,"dimensions":[2]}`,
		},
	}
	for _, row := range rows {
		if err := db.Create(row).Error; err != nil {
			t.Fatalf("failed to seed %T: %v", row, err)
		}
	}

	engine := &embeddingSingleReadEngine{embeddingSampleEngine: embeddingSampleEngine{
		chunks: map[string]map[string]interface{}{
			"chunk-1": {
				"id":           "chunk-1",
				"doc_id":       "doc-1",
				"docnm_kwd":    "Title",
				"title_tks":    "Title",
				"content_ltks": "Content",
				"q_2_vec":      []interface{}{1.0, 0.0},
			},
		},
	}}
	service := NewDatasetService()
	service.docEngine = engine
	checkNum := 1

	response, code, err := service.CheckEmbedding(t.Context(), "user-1", "dataset-1", &serviceModule.CheckEmbeddingRequest{
		EmbeddingID: "candidate-model",
		CheckNum:    &checkNum,
	})
	if err != nil {
		t.Fatalf("CheckEmbedding() error = %v", err)
	}
	if code != common.CodeSuccess {
		t.Fatalf("code = %v, want %v", code, common.CodeSuccess)
	}
	if response == nil || response.Summary.Valid != 1 {
		t.Fatalf("response = %#v, want one valid comparison", response)
	}
	requestPathMu.Lock()
	actualRequestPath := requestPath
	requestPathMu.Unlock()
	if actualRequestPath != "/embeddings" {
		t.Fatalf("provider path = %q, want /embeddings", actualRequestPath)
	}
	if engine.getCalls != 1 {
		t.Fatalf("GetChunk calls = %d, want 1", engine.getCalls)
	}
}

func TestCheckEmbeddingClassifiesValidSampleShortfallAsDataError(t *testing.T) {
	service := setupEmbeddingCheckService(t, "", &embeddingSampleEngine{
		chunks: map[string]map[string]interface{}{
			"invalid": {
				"id": "invalid",
			},
			"valid": {
				"id":           "valid",
				"title_tks":    "Title",
				"content_ltks": "Content",
				"q_2_vec":      []interface{}{1.0, 0.0},
			},
		},
	})
	checkNum := 2

	response, code, err := service.CheckEmbedding(
		t.Context(),
		"user-1",
		"dataset-1",
		&serviceModule.CheckEmbeddingRequest{
			EmbeddingID: "candidate-model",
			CheckNum:    &checkNum,
		},
	)
	if err == nil || !strings.Contains(err.Error(), "found 1 of 2") {
		t.Fatalf("CheckEmbedding() error = %v, want valid sample shortfall", err)
	}
	if code != common.CodeDataError {
		t.Fatalf("code = %v, want %v", code, common.CodeDataError)
	}
	if response != nil {
		t.Fatalf("response = %#v, want nil", response)
	}
}

func TestCheckEmbeddingRejectsMalformedProviderVectors(t *testing.T) {
	tests := []struct {
		name     string
		response string
		wantErr  string
	}{
		{
			name:     "empty embedding list",
			response: `{"data":[]}`,
			wantErr:  "embedding response count",
		},
		{
			name:     "empty embedding vector",
			response: `{"data":[{"embedding":[],"index":0}]}`,
			wantErr:  "empty embedding vector",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(test.response))
			}))
			t.Cleanup(server.Close)

			service := setupEmbeddingCheckService(t, server.URL, &embeddingSampleEngine{
				chunks: map[string]map[string]interface{}{
					"chunk-1": {
						"id":           "chunk-1",
						"content_ltks": "Content",
						"q_2_vec":      []interface{}{1.0, 0.0},
					},
				},
			})
			checkNum := 1

			response, code, err := service.CheckEmbedding(
				t.Context(),
				"user-1",
				"dataset-1",
				&serviceModule.CheckEmbeddingRequest{
					EmbeddingID: "candidate-model",
					CheckNum:    &checkNum,
				},
			)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("CheckEmbedding() error = %v, want %q", err, test.wantErr)
			}
			if code != common.CodeServerError {
				t.Fatalf("code = %v, want %v", code, common.CodeServerError)
			}
			if response != nil {
				t.Fatalf("response = %#v, want nil", response)
			}
		})
	}
}

func TestDatasetEncodeEmbeddingRejectsNonFiniteVectors(t *testing.T) {
	tests := []struct {
		name  string
		value float64
	}{
		{name: "NaN", value: math.NaN()},
		{name: "positive infinity", value: math.Inf(1)},
		{name: "negative infinity", value: math.Inf(-1)},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			driver := &embeddingProtocolDriver{
				vectors: [][]float64{{1, test.value}},
			}
			modelName := "candidate"
			model := &modelModule.EmbeddingModel{
				ModelDriver: driver,
				ModelName:   &modelName,
				APIConfig:   &modelModule.APIConfig{},
			}

			vectors, err := datasetEncodeEmbedding(
				t.Context(),
				model,
				[]string{"Content"},
			)
			if err == nil || !strings.Contains(err.Error(), "non-finite") {
				t.Fatalf(
					"datasetEncodeEmbedding() error = %v, want non-finite vector error",
					err,
				)
			}
			if vectors != nil {
				t.Fatalf("vectors = %#v, want nil", vectors)
			}
		})
	}
}

func TestCheckEmbeddingRejectsProviderDimensionMismatch(t *testing.T) {
	tests := []struct {
		name     string
		vectors  [][]float64
		stored   []interface{}
		wantCode common.ErrorCode
		wantErr  string
	}{
		{
			name:     "title and content differ",
			vectors:  [][]float64{{1, 0}, {1, 0, 0}},
			stored:   []interface{}{1.0, 0.0},
			wantCode: common.CodeServerError,
			wantErr:  "embedding response dimensions do not match",
		},
		{
			name:     "candidate and stored differ",
			vectors:  [][]float64{{1, 0, 0}, {1, 0, 0}},
			stored:   []interface{}{1.0, 0.0},
			wantCode: common.CodeDataError,
			wantErr:  "does not match stored dimension",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			driver := &embeddingProtocolDriver{vectors: test.vectors}
			providerInfo := dao.GetModelProviderManager().FindProvider("OpenAI")
			if providerInfo == nil {
				t.Fatal("OpenAI provider metadata missing")
			}
			originalDriver := providerInfo.ModelDriver
			providerInfo.ModelDriver = driver
			t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

			service := setupEmbeddingCheckService(t, "", &embeddingSampleEngine{
				chunks: map[string]map[string]interface{}{
					"chunk-1": {
						"id":           "chunk-1",
						"title_tks":    "Title",
						"content_ltks": "Content",
						"q_2_vec":      test.stored,
					},
				},
			})
			checkNum := 1

			response, code, err := service.CheckEmbedding(
				t.Context(),
				"user-1",
				"dataset-1",
				&serviceModule.CheckEmbeddingRequest{
					EmbeddingID: "candidate-model",
					CheckNum:    &checkNum,
				},
			)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf(
					"CheckEmbedding() error = %v, want %q",
					err,
					test.wantErr,
				)
			}
			if code != test.wantCode {
				t.Fatalf("code = %v, want %v", code, test.wantCode)
			}
			if response != nil {
				t.Fatalf("response = %#v, want nil", response)
			}
		})
	}
}

func TestCheckEmbeddingRejectsSamplesWithoutEmbeddableText(t *testing.T) {
	service := setupEmbeddingCheckService(t, "", &embeddingSampleEngine{
		chunks: map[string]map[string]interface{}{
			"empty-text": {
				"id":      "empty-text",
				"q_2_vec": []interface{}{1.0, 0.0},
			},
		},
	})
	checkNum := 1

	response, code, err := service.CheckEmbedding(
		t.Context(),
		"user-1",
		"dataset-1",
		&serviceModule.CheckEmbeddingRequest{
			EmbeddingID: "candidate-model",
			CheckNum:    &checkNum,
		},
	)
	if err == nil || !strings.Contains(err.Error(), "found 0 of 1") {
		t.Fatalf("CheckEmbedding() error = %v, want valid sample shortfall", err)
	}
	if code != common.CodeDataError {
		t.Fatalf("code = %v, want %v", code, common.CodeDataError)
	}
	if response != nil {
		t.Fatalf("response = %#v, want nil", response)
	}
}

func TestCheckEmbeddingClassifiesEngineFailuresAsServerError(t *testing.T) {
	tests := []struct {
		name    string
		engine  engine.DocEngine
		wantErr string
	}{
		{
			name: "search error",
			engine: &embeddingSearchFailureEngine{
				err: errors.New("search unavailable"),
			},
			wantErr: "search unavailable",
		},
		{
			name:    "nil search response",
			engine:  &embeddingSearchFailureEngine{},
			wantErr: "empty search response",
		},
		{
			name: "nil chunk response",
			engine: &embeddingGetChunkFailureEngine{
				embeddingSampleEngine: embeddingSampleEngine{
					chunks: map[string]map[string]interface{}{
						"chunk-1": {"id": "chunk-1"},
					},
				},
			},
			wantErr: "chunk not found",
		},
		{
			name: "malformed chunk response",
			engine: &embeddingGetChunkFailureEngine{
				embeddingSampleEngine: embeddingSampleEngine{
					chunks: map[string]map[string]interface{}{
						"chunk-1": {"id": "chunk-1"},
					},
				},
				response: []string{"not", "a", "chunk"},
			},
			wantErr: "malformed chunk response",
		},
		{
			name:    "empty counted page",
			engine:  &embeddingMalformedPageEngine{},
			wantErr: "returned no chunk",
		},
		{
			name: "counted page missing chunk ID",
			engine: &embeddingMalformedPageEngine{
				chunk: map[string]interface{}{"content": "missing ID"},
			},
			wantErr: "missing chunk ID",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			service := setupEmbeddingCheckService(t, "", test.engine)
			checkNum := 1

			response, code, err := service.CheckEmbedding(
				t.Context(),
				"user-1",
				"dataset-1",
				&serviceModule.CheckEmbeddingRequest{
					EmbeddingID: "candidate-model",
					CheckNum:    &checkNum,
				},
			)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf(
					"CheckEmbedding() error = %v, want %q",
					err,
					test.wantErr,
				)
			}
			if code != common.CodeServerError {
				t.Fatalf("code = %v, want %v", code, common.CodeServerError)
			}
			if response != nil {
				t.Fatalf("response = %#v, want nil", response)
			}
		})
	}
}

func TestSampleRandomChunksWithVectorsReplacesInvalidSamples(t *testing.T) {
	engine := &embeddingReplacementEngine{embeddingSampleEngine: embeddingSampleEngine{
		chunks: map[string]map[string]interface{}{
			"invalid": {
				"id":                  "invalid",
				"content_with_weight": "no vector",
			},
			"valid": {
				"id":                  "valid",
				"doc_id":              "doc-1",
				"content_with_weight": "has vector",
				"content_ltks":        "has vector",
				"q_3_vec":             []interface{}{0.25, -0.5, 1.0},
			},
		},
	}}
	service := &DatasetService{docEngine: engine}

	samples, err := service.sampleRandomChunksWithVectors(t.Context(), "tenant-1", "dataset-1", 1)
	if err != nil {
		t.Fatalf("sampleRandomChunksWithVectors() error = %v", err)
	}
	if len(samples) != 1 || samples[0].ChunkID != "valid" {
		t.Fatalf("samples = %#v, want one valid replacement", samples)
	}
	if engine.searchCalls != 2 {
		t.Fatalf("search calls = %d, want 2", engine.searchCalls)
	}
}

func TestSampleRandomChunksWithVectorsRejectsInsufficientValidSamples(t *testing.T) {
	service := &DatasetService{docEngine: &embeddingSampleEngine{
		chunks: map[string]map[string]interface{}{
			"invalid": {
				"id":                  "invalid",
				"content_with_weight": "no vector",
			},
			"valid": {
				"id":                  "valid",
				"doc_id":              "doc-1",
				"content_with_weight": "has vector",
				"content_ltks":        "has vector",
				"q_3_vec":             []interface{}{0.25, -0.5, 1.0},
			},
		},
	}}

	samples, err := service.sampleRandomChunksWithVectors(t.Context(), "tenant-1", "dataset-1", 2)
	if err == nil || !strings.Contains(err.Error(), "found 1 of 2") {
		t.Fatalf("sampleRandomChunksWithVectors() error = %v, want valid sample shortfall", err)
	}
	if samples != nil {
		t.Fatalf("samples = %#v, want nil on sample shortfall", samples)
	}
}

type embeddingByTextDriver struct {
	modelModule.ModelDriver
	vectors map[string][]float64
}

func (d *embeddingByTextDriver) Embed(
	_ context.Context,
	_ *string,
	request modelModule.EmbedRequest,
	_ *modelModule.APIConfig,
	_ *modelModule.EmbeddingConfig,
	_ *common.ModelUsage,
) ([]modelModule.EmbeddingData, error) {
	result := make([]modelModule.EmbeddingData, 0, len(request.Texts))
	for _, text := range request.Texts {
		vector, ok := d.vectors[text]
		if !ok {
			return nil, errors.New("unexpected embedding text")
		}
		result = append(result, modelModule.EmbeddingData{Embedding: vector})
	}
	return result, nil
}

func TestCheckEmbeddingRejectsZeroIndexedChunks(t *testing.T) {
	service := setupEmbeddingCheckService(t, "", &embeddingSearchFailureEngine{
		response: &enginetypes.SearchResult{Total: 0},
	})
	checkNum := 1

	response, code, err := service.CheckEmbedding(
		t.Context(),
		"user-1",
		"dataset-1",
		&serviceModule.CheckEmbeddingRequest{
			EmbeddingID: "candidate-model",
			CheckNum:    &checkNum,
		},
	)
	if err == nil || !errors.Is(err, errInsufficientEmbeddingSamples) {
		t.Fatalf("CheckEmbedding() error = %v, want insufficient samples error", err)
	}
	if code != common.CodeDataError {
		t.Fatalf("code = %v, want %v", code, common.CodeDataError)
	}
	if response != nil {
		t.Fatalf("response = %#v, want nil", response)
	}
}

func TestCheckEmbeddingRejectsNegativeSearchTotal(t *testing.T) {
	service := setupEmbeddingCheckService(t, "", &embeddingSearchFailureEngine{
		response: &enginetypes.SearchResult{Total: -1},
	})
	checkNum := 1

	response, code, err := service.CheckEmbedding(
		t.Context(),
		"user-1",
		"dataset-1",
		&serviceModule.CheckEmbeddingRequest{
			EmbeddingID: "candidate-model",
			CheckNum:    &checkNum,
		},
	)
	if err == nil || !strings.Contains(err.Error(), "negative total") {
		t.Fatalf("CheckEmbedding() error = %v, want negative-total error", err)
	}
	if code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}
	if response != nil {
		t.Fatalf("response = %#v, want nil", response)
	}
}

func TestCheckEmbeddingRejectsStoredDimensionDriftAcrossSamples(t *testing.T) {
	driver := &embeddingByTextDriver{
		vectors: map[string][]float64{
			"two":   {1, 0},
			"three": {1, 0, 0},
		},
	}
	providerInfo := dao.GetModelProviderManager().FindProvider("OpenAI")
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = driver
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	service := setupEmbeddingCheckService(t, "", &embeddingSampleEngine{
		chunks: map[string]map[string]interface{}{
			"chunk-2": {
				"id":           "chunk-2",
				"content_ltks": "two",
				"q_2_vec":      []interface{}{1.0, 0.0},
			},
			"chunk-3": {
				"id":           "chunk-3",
				"content_ltks": "three",
				"q_3_vec":      []interface{}{1.0, 0.0, 0.0},
			},
		},
	})
	checkNum := 2

	response, code, err := service.CheckEmbedding(
		t.Context(),
		"user-1",
		"dataset-1",
		&serviceModule.CheckEmbeddingRequest{
			EmbeddingID: "candidate-model",
			CheckNum:    &checkNum,
		},
	)
	if err == nil || !strings.Contains(err.Error(), "stored vector dimension changed") {
		t.Fatalf("CheckEmbedding() error = %v, want stored-dimension drift error", err)
	}
	if code != common.CodeDataError {
		t.Fatalf("code = %v, want %v", code, common.CodeDataError)
	}
	if response != nil {
		t.Fatalf("response = %#v, want nil", response)
	}
}

func TestCheckEmbeddingRejectsProviderDimensionDriftAcrossSamples(t *testing.T) {
	driver := &embeddingByTextDriver{
		vectors: map[string][]float64{
			"first":  {1, 0},
			"second": {1, 0, 0},
		},
	}
	providerInfo := dao.GetModelProviderManager().FindProvider("OpenAI")
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = driver
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	service := setupEmbeddingCheckService(t, "", &embeddingSampleEngine{
		chunks: map[string]map[string]interface{}{
			"chunk-first": {
				"id":           "chunk-first",
				"content_ltks": "first",
				"q_2_vec":      []interface{}{1.0, 0.0},
			},
			"chunk-second": {
				"id":           "chunk-second",
				"content_ltks": "second",
				"q_2_vec":      []interface{}{1.0, 0.0},
			},
		},
	})
	checkNum := 2

	response, code, err := service.CheckEmbedding(
		t.Context(),
		"user-1",
		"dataset-1",
		&serviceModule.CheckEmbeddingRequest{
			EmbeddingID: "candidate-model",
			CheckNum:    &checkNum,
		},
	)
	if err == nil || !strings.Contains(err.Error(), "provider embedding dimension changed") {
		t.Fatalf("CheckEmbedding() error = %v, want provider-dimension drift error", err)
	}
	if code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}
	if response != nil {
		t.Fatalf("response = %#v, want nil", response)
	}
}
