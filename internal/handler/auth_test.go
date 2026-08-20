//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

package handler

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"ragflow/internal/common"
	"ragflow/internal/entity"
	"ragflow/internal/server/local"
)

type fixedBetaRateLimiter struct {
	allowed bool
	err     error
}

func (f fixedBetaRateLimiter) Allow(context.Context, string) (bool, error) {
	return f.allowed, f.err
}

type fixedUserTokenResolver struct {
	user *entity.User
}

func (f fixedUserTokenResolver) GetUserByToken(context.Context, string) (*entity.User, common.ErrorCode, error) {
	return f.user, common.CodeSuccess, nil
}

func (f fixedUserTokenResolver) GetUserByAPIToken(context.Context, string) (*entity.User, common.ErrorCode, error) {
	return nil, common.CodeUnauthorized, errors.New("not an API token")
}

func (f fixedUserTokenResolver) GetUserByBetaAPIToken(context.Context, string) (*entity.User, common.ErrorCode, error) {
	return nil, common.CodeUnauthorized, errors.New("not a beta token")
}

func (f fixedUserTokenResolver) GetAPITokenByBeta(context.Context, string) (*entity.APIToken, error) {
	return nil, errors.New("not a beta token")
}

func TestAuthMiddleware_AllowsSuperuserOwnTenantSession(t *testing.T) {
	gin.SetMode(gin.TestMode)
	previous := local.GetAdminStatus()
	local.SetAdminStatus(0, "")
	t.Cleanup(func() { local.SetAdminStatus(previous.Status, previous.Reason) })

	isSuperuser := true
	handler := &AuthHandler{userService: fixedUserTokenResolver{user: &entity.User{
		ID:          "owner-id",
		Email:       "owner@example.test",
		IsSuperuser: &isSuperuser,
	}}}
	recorder := httptest.NewRecorder()
	context, _ := gin.CreateTestContext(recorder)
	context.Request = httptest.NewRequest(http.MethodGet, "/api/v1/tenants/owner-id", nil)
	context.Request.Header.Set("Authorization", "signed-session")

	handler.AuthMiddleware()(context)

	if context.IsAborted() {
		t.Fatalf("superuser tenant session was rejected: status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if userID, exists := context.Get("user_id"); !exists || userID != "owner-id" {
		t.Fatalf("user_id = %v, exists=%v", userID, exists)
	}
}

// TestBetaAuthMiddleware_MissingHeader pins the no-header branch —
// the middleware must short-circuit with 401/CodeUnauthorized and
// must not call into UserService. The other branches (regular JWT
// and beta token) require a live DB to resolve, so they are covered
// by the cross-cutting TestBotRoutes_RequireAuth criterion in
// bot_test.go.
func TestBetaAuthMiddleware_MissingHeader(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ah := &AuthHandler{userService: nil}
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodGet, "/", nil)

	mw := ah.BetaAuthMiddleware()
	mw(c)

	if !c.IsAborted() {
		t.Fatalf("context not aborted, want aborted (no Authorization header)")
	}
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	// Beta endpoints follow Python's login_required(auth_types=AUTH_BETA):
	// auth failures are business errors (code 102), not HTTP 401.
	var resp struct {
		Code    common.ErrorCode `json:"code"`
		Message string           `json:"message"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v; body = %s", err, rec.Body.String())
	}
	if resp.Code != common.CodeDataError {
		t.Errorf("code = %d, want %d; body = %s",
			resp.Code, common.CodeDataError, rec.Body.String())
	}
	if resp.Message != "Authorization is not valid!" {
		t.Errorf("message = %q; body = %s", resp.Message, rec.Body.String())
	}
}

func TestBetaAuthMiddleware_RateLimitExceeded(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ah := &AuthHandler{betaRateLimiter: fixedBetaRateLimiter{allowed: false}}
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/mcp", nil)
	c.Request.Header.Set("Authorization", "public-token-value")

	ah.BetaAuthMiddleware()(c)

	if !c.IsAborted() || rec.Code != http.StatusTooManyRequests {
		t.Fatalf("aborted = %v, status = %d, body = %s", c.IsAborted(), rec.Code, rec.Body.String())
	}
	if rec.Body.String() == "" || containsSecret(rec.Body.String(), "public-token-value") {
		t.Fatalf("rate-limit response leaked credential: %s", rec.Body.String())
	}
}

func TestBetaAuthMiddleware_RateLimiterFailureFailsClosed(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ah := &AuthHandler{betaRateLimiter: fixedBetaRateLimiter{err: errors.New("redis down")}}
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/mcp", nil)
	c.Request.Header.Set("Authorization", "public-token-value")

	ah.BetaAuthMiddleware()(c)

	if !c.IsAborted() || rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("aborted = %v, status = %d, body = %s", c.IsAborted(), rec.Code, rec.Body.String())
	}
}

func containsSecret(body, secret string) bool {
	return len(secret) > 0 && len(body) >= len(secret) && stringContains(body, secret)
}

func stringContains(value, fragment string) bool {
	for i := 0; i+len(fragment) <= len(value); i++ {
		if value[i:i+len(fragment)] == fragment {
			return true
		}
	}
	return false
}
