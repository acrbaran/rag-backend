package handler

import (
	"net/http"
	"testing"

	"ragflow/internal/service"
)

func TestProviderHandlerChatToModelAcceptsModelIDOnlyRequest(t *testing.T) {
	db := setupProviderHandlerTestDB(t)
	useProviderHandlerTestDB(t, db)

	ctx, recorder := newProviderHandlerRequest(t, map[string]interface{}{
		"model_id": "missing-model",
		"messages": []map[string]interface{}{{
			"role":    "user",
			"content": "test",
		}},
		"stream": false,
	})

	NewProviderHandler(nil, service.NewModelProviderService()).ChatToModel(ctx)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body=%s", recorder.Code, http.StatusOK, recorder.Body.String())
	}
}
