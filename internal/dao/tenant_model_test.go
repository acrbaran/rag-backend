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

package dao

import (
	"errors"
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"

	"ragflow/internal/entity"
)

func setupTenantModelDAOTestDB(t *testing.T) *gorm.DB {
	t.Helper()
	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{TranslateError: true})
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	if err = db.AutoMigrate(
		&entity.TenantModelProvider{},
		&entity.TenantModelInstance{},
		&entity.TenantModel{},
	); err != nil {
		t.Fatalf("failed to migrate tenant_model: %v", err)
	}
	return db
}

func useTenantModelDAOTestDB(t *testing.T, db *gorm.DB) {
	t.Helper()
	orig := DB
	DB = db
	t.Cleanup(func() { DB = orig })
}

func seedTenantModel(t *testing.T, db *gorm.DB, model *entity.TenantModel) {
	t.Helper()
	if err := db.Create(model).Error; err != nil {
		t.Fatalf("failed to seed tenant model: %v", err)
	}
}

func TestTenantModelDAOUpdateByIDAndScopeUpdatesOnlyMatchingModel(t *testing.T) {
	db := setupTenantModelDAOTestDB(t)
	useTenantModelDAOTestDB(t, db)

	seedTenantModel(t, db, &entity.TenantModel{ID: "model-update", ModelName: "m", ModelType: int(entity.ModelTypeChat), ProviderID: "provider-1", InstanceID: "instance-1", Status: "active", Extra: "{}"})

	updates := map[string]interface{}{
		"status":     "inactive",
		"model_type": int(entity.ModelTypeEmbedding),
		"extra":      `{"max_tokens":1024}`,
	}
	rows, err := NewTenantModelDAO().UpdateByIDAndScope(
		t.Context(),
		db,
		"model-update",
		"provider-1",
		"instance-1",
		updates,
	)
	if err != nil {
		t.Fatalf("UpdateByIDAndScope() error = %v", err)
	}
	if rows != 1 {
		t.Fatalf("rows = %d, want 1", rows)
	}

	var got entity.TenantModel
	if err = db.Where("id = ?", "model-update").First(&got).Error; err != nil {
		t.Fatalf("failed to reload model: %v", err)
	}
	if got.Status != "inactive" || got.ModelType != int(entity.ModelTypeEmbedding) || got.Extra != `{"max_tokens":1024}` {
		t.Fatalf("updated model = %#v", got)
	}

	rows, err = NewTenantModelDAO().UpdateByIDAndScope(
		t.Context(),
		db,
		"model-update",
		"provider-1",
		"wrong-instance",
		map[string]interface{}{"status": "active"},
	)
	if err != nil {
		t.Fatalf("UpdateByIDAndScope() wrong scope error = %v", err)
	}
	if rows != 0 {
		t.Fatalf("wrong-scope rows = %d, want 0", rows)
	}
}

func TestTenantModelDAODeleteByModelIDAndScopeDeletesOnlyMatchingModel(t *testing.T) {
	db := setupTenantModelDAOTestDB(t)
	useTenantModelDAOTestDB(t, db)

	seedTenantModel(t, db, &entity.TenantModel{ID: "model-delete", ModelName: "m", ModelType: int(entity.ModelTypeChat), ProviderID: "provider-1", InstanceID: "instance-1", Status: "active"})
	seedTenantModel(t, db, &entity.TenantModel{ID: "model-keep", ModelName: "m", ModelType: int(entity.ModelTypeChat), ProviderID: "provider-1", InstanceID: "instance-2", Status: "active"})

	ctx := t.Context()
	rows, err := NewTenantModelDAO().DeleteByModelIDAndProviderIDAndInstanceID(ctx, db, "model-delete", "provider-1", "instance-1")
	if err != nil {
		t.Fatalf("DeleteByModelIDAndProviderIDAndInstanceID() error = %v", err)
	}
	if rows != 1 {
		t.Fatalf("rows = %d, want 1", rows)
	}

	var count int64
	if err = db.Model(&entity.TenantModel{}).Where("id = ?", "model-delete").Count(&count).Error; err != nil {
		t.Fatalf("count deleted model: %v", err)
	}
	if count != 0 {
		t.Fatalf("deleted model count = %d, want 0", count)
	}
	if err = db.Model(&entity.TenantModel{}).Where("id = ?", "model-keep").Count(&count).Error; err != nil {
		t.Fatalf("count kept model: %v", err)
	}
	if count != 1 {
		t.Fatalf("kept model count = %d, want 1", count)
	}
}

func TestTenantModelDAOUpdateStatusWithUpdateByIDAndScope(t *testing.T) {
	db := setupTenantModelDAOTestDB(t)
	useTenantModelDAOTestDB(t, db)

	seedTenantModel(t, db, &entity.TenantModel{ID: "model-status", ModelName: "m", ModelType: int(entity.ModelTypeChat), ProviderID: "provider-1", InstanceID: "instance-1", Status: "active"})

	ctx := t.Context()
	rows, err := NewTenantModelDAO().UpdateByIDAndScope(
		ctx,
		db,
		"model-status",
		"provider-1",
		"instance-1",
		map[string]interface{}{"status": "inactive"},
	)
	if err != nil {
		t.Fatalf("UpdateByIDAndScope() error = %v", err)
	}
	if rows != 1 {
		t.Fatalf("rows = %d, want 1", rows)
	}

	var got entity.TenantModel
	if err = db.Where("id = ?", "model-status").First(&got).Error; err != nil {
		t.Fatalf("failed to reload model: %v", err)
	}
	if got.Status != "inactive" {
		t.Fatalf("status = %q, want inactive", got.Status)
	}

	rows, err = NewTenantModelDAO().UpdateByIDAndScope(
		ctx,
		db,
		"model-status",
		"provider-1",
		"wrong-instance",
		map[string]interface{}{"status": "active"},
	)
	if err != nil {
		t.Fatalf("UpdateByIDAndScope() wrong scope error = %v", err)
	}
	if rows != 0 {
		t.Fatalf("wrong-scope rows = %d, want 0", rows)
	}
}

func TestTenantModelProviderDAOGetByIDAndTenantIDForUpdateUsesTenantScope(t *testing.T) {
	db := setupTenantModelDAOTestDB(t)
	providers := []*entity.TenantModelProvider{
		{ID: "provider-1", ProviderName: "OpenAI", TenantID: "tenant-1"},
		{ID: "provider-2", ProviderName: "OpenAI", TenantID: "tenant-2"},
	}
	for _, provider := range providers {
		if err := db.Create(provider).Error; err != nil {
			t.Fatalf("failed to seed provider: %v", err)
		}
	}

	got, err := NewTenantModelProviderDAO().GetByIDAndTenantIDForUpdate(
		t.Context(),
		db,
		"provider-1",
		"tenant-1",
	)
	if err != nil {
		t.Fatalf("GetByIDAndTenantIDForUpdate() error = %v", err)
	}
	if got.ID != "provider-1" {
		t.Fatalf("provider ID = %q, want provider-1", got.ID)
	}

	_, err = NewTenantModelProviderDAO().GetByIDAndTenantIDForUpdate(
		t.Context(),
		db,
		"provider-1",
		"tenant-2",
	)
	if !errors.Is(err, gorm.ErrRecordNotFound) {
		t.Fatalf("wrong-tenant error = %v, want record not found", err)
	}
}

func TestTenantModelInstanceDAOUpdateByIDAndProviderIDReportsScopedRows(t *testing.T) {
	db := setupTenantModelDAOTestDB(t)
	instances := []*entity.TenantModelInstance{
		{ID: "instance-1", InstanceName: "first", ProviderID: "provider-1", APIKey: "key", Status: "active", Extra: "{}"},
		{ID: "instance-2", InstanceName: "second", ProviderID: "provider-2", APIKey: "key", Status: "active", Extra: "{}"},
	}
	for _, instance := range instances {
		if err := db.Create(instance).Error; err != nil {
			t.Fatalf("failed to seed instance: %v", err)
		}
	}

	rows, err := NewTenantModelInstanceDAO().UpdateByIDAndProviderID(
		t.Context(),
		db,
		"instance-1",
		"provider-1",
		map[string]interface{}{"status": "inactive"},
	)
	if err != nil {
		t.Fatalf("UpdateByIDAndProviderID() error = %v", err)
	}
	if rows != 1 {
		t.Fatalf("rows = %d, want 1", rows)
	}

	rows, err = NewTenantModelInstanceDAO().UpdateByIDAndProviderID(
		t.Context(),
		db,
		"instance-1",
		"provider-2",
		map[string]interface{}{"status": "active"},
	)
	if err != nil {
		t.Fatalf("wrong-scope update error = %v", err)
	}
	if rows != 0 {
		t.Fatalf("wrong-scope rows = %d, want 0", rows)
	}

	var got entity.TenantModelInstance
	if err := db.Where("id = ?", "instance-1").First(&got).Error; err != nil {
		t.Fatalf("failed to reload instance: %v", err)
	}
	if got.Status != "inactive" {
		t.Fatalf("status = %q, want inactive", got.Status)
	}
}

func TestTenantModelDAODeleteByIDsAndScopeReportsScopedRows(t *testing.T) {
	db := setupTenantModelDAOTestDB(t)
	seedTenantModel(t, db, &entity.TenantModel{ID: "model-delete-scoped", ModelName: "first", ModelType: int(entity.ModelTypeChat), ProviderID: "provider-1", InstanceID: "instance-1", Status: "active", Extra: "{}"})
	seedTenantModel(t, db, &entity.TenantModel{ID: "model-keep-scoped", ModelName: "second", ModelType: int(entity.ModelTypeChat), ProviderID: "provider-1", InstanceID: "instance-2", Status: "active", Extra: "{}"})

	rows, err := NewTenantModelDAO().DeleteByIDsAndScope(
		t.Context(),
		db,
		[]string{"model-delete-scoped", "model-keep-scoped"},
		"provider-1",
		"instance-1",
	)
	if err != nil {
		t.Fatalf("DeleteByIDsAndScope() error = %v", err)
	}
	if rows != 1 {
		t.Fatalf("rows = %d, want 1", rows)
	}

	var count int64
	if err := db.Model(&entity.TenantModel{}).
		Where("id = ?", "model-keep-scoped").
		Count(&count).Error; err != nil {
		t.Fatalf("count kept model: %v", err)
	}
	if count != 1 {
		t.Fatalf("kept model count = %d, want 1", count)
	}
}
