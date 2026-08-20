//go:build integration

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
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	drivermysql "github.com/go-sql-driver/mysql"
	gormmysql "gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"ragflow/internal/entity"
)

const tenantModelMigrationTestDSNEnv = "RAGFLOW_MYSQL_TEST_DSN"

type tenantModelMigrationFixture struct {
	ID         string
	ModelName  string
	ProviderID string
	InstanceID string
	ModelType  string
	Status     string
	Extra      string
}

func TestMySQLDatabaseNameFromDSNMatchesDriverParsing(t *testing.T) {
	tests := []struct {
		name    string
		dsn     string
		want    string
		wantErr bool
	}{
		{name: "TCP", dsn: "user:password@tcp(localhost:3306)/ragflow_migration_test?parseTime=true", want: "ragflow_migration_test"},
		{name: "Unix socket", dsn: "user:password@unix(/tmp/mysql.sock)/ragflow_migration_test", want: "ragflow_migration_test"},
		{name: "Percent escape stays literal", dsn: "user:password@tcp(localhost:3306)/prod%5Ftest", want: "prod%5Ftest"},
		{name: "Missing database", dsn: "user:password@tcp(localhost:3306)/", wantErr: true},
		{name: "Malformed", dsn: "not-a-dsn", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := mysqlDatabaseNameFromDSN(tt.dsn)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("mysqlDatabaseNameFromDSN(%q) = %q, want error", tt.dsn, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("mysqlDatabaseNameFromDSN(%q) error = %v", tt.dsn, err)
			}
			if got != tt.want {
				t.Fatalf("mysqlDatabaseNameFromDSN(%q) = %q, want %q", tt.dsn, got, tt.want)
			}
		})
	}
}

func TestRunRequiredStartupMigrationsCancellationReleasesLockMySQL(t *testing.T) {
	db := openTenantModelMigrationTestDB(t)
	var databaseName string
	if err := db.Raw("SELECT DATABASE()").Scan(&databaseName).Error; err != nil {
		t.Fatalf("read MySQL test database name: %v", err)
	}

	if err := db.Exec("DROP TABLE IF EXISTS tenant_model").Error; err != nil {
		t.Fatalf("drop tenant_model test table: %v", err)
	}
	t.Cleanup(func() {
		if err := db.Exec("DROP TABLE IF EXISTS tenant_model").Error; err != nil {
			t.Errorf("cleanup tenant_model test table: %v", err)
		}
	})
	if err := db.Exec(`
		CREATE TABLE tenant_model (
			id VARCHAR(32) NOT NULL PRIMARY KEY,
			model_name VARCHAR(128) NOT NULL,
			provider_id VARCHAR(32) NOT NULL,
			instance_id VARCHAR(32) NOT NULL,
			model_type VARCHAR(64) NULL
		)
	`).Error; err != nil {
		t.Fatalf("create tenant_model test table: %v", err)
	}
	if err := db.Exec(
		"INSERT INTO tenant_model (id, model_name, provider_id, instance_id, model_type) VALUES ('blocked', 'blocked-model', 'provider', 'instance', 'chat')",
	).Error; err != nil {
		t.Fatalf("insert blocked tenant_model row: %v", err)
	}

	blocker := db.Begin()
	if blocker.Error != nil {
		t.Fatalf("begin blocker transaction: %v", blocker.Error)
	}
	blockerActive := true
	t.Cleanup(func() {
		if blockerActive {
			if err := blocker.Rollback().Error; err != nil {
				t.Errorf("cleanup blocker transaction: %v", err)
			}
		}
	})
	if err := blocker.Exec(
		"SELECT id FROM tenant_model WHERE id = 'blocked' FOR UPDATE",
	).Error; err != nil {
		t.Fatalf("lock tenant_model row: %v", err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	migrationErr := make(chan error, 1)
	go func() {
		migrationErr <- RunRequiredStartupMigrations(ctx, db, false)
	}()

	lockName := requiredStartupMigrationLockName(databaseName)
	lockDeadline := time.Now().Add(5 * time.Second)
	for {
		var owner *int64
		if err := db.Raw(
			"SELECT IS_USED_LOCK(?) AS lock_owner",
			lockName,
		).Scan(&owner).Error; err != nil {
			cancel()
			t.Fatalf("inspect advisory lock owner: %v", err)
		}
		if owner != nil {
			break
		}
		if time.Now().After(lockDeadline) {
			cancel()
			t.Fatal("required startup migration did not acquire advisory lock")
		}
		time.Sleep(10 * time.Millisecond)
	}

	cancel()
	err := <-migrationErr
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("RunRequiredStartupMigrations() error = %v, want context canceled", err)
	}
	if strings.Contains(err.Error(), "release advisory lock") {
		t.Fatalf("RunRequiredStartupMigrations() error = %v, want cancellation without misleading release error", err)
	}
	if err := blocker.Rollback().Error; err != nil {
		t.Fatalf("rollback blocker transaction: %v", err)
	}
	blockerActive = false

	if err := db.Connection(func(conn *gorm.DB) error {
		var acquired *int64
		if err := conn.Raw(
			"SELECT GET_LOCK(?, 1)",
			lockName,
		).Scan(&acquired).Error; err != nil {
			return fmt.Errorf("reacquire advisory lock after cancellation: %w", err)
		}
		if acquired == nil || *acquired != 1 {
			return fmt.Errorf("reacquire advisory lock after cancellation = %v, want 1", acquired)
		}
		var released *int64
		if err := conn.Raw(
			"SELECT RELEASE_LOCK(?) AS lock_result",
			lockName,
		).Scan(&released).Error; err != nil {
			return fmt.Errorf("release reacquired advisory lock: %w", err)
		}
		if released == nil || *released != 1 {
			return fmt.Errorf("release reacquired advisory lock = %v, want 1", released)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestRunRequiredStartupMigrationsBootstrapsMissingTenantModelMySQL(t *testing.T) {
	db := openTenantModelMigrationTestDB(t)

	if err := db.Exec("DROP TABLE IF EXISTS tenant_model").Error; err != nil {
		t.Fatalf("drop tenant_model test table: %v", err)
	}
	t.Cleanup(func() {
		if err := db.Exec("DROP TABLE IF EXISTS tenant_model").Error; err != nil {
			t.Errorf("cleanup tenant_model test table: %v", err)
		}
	})

	if err := RunRequiredStartupMigrations(t.Context(), db, true); err != nil {
		t.Fatalf("RunRequiredStartupMigrations() error = %v, want fresh bootstrap", err)
	}
	assertTenantModelMigrationColumn(t, db)
}

func TestTenantModelTypeStartupMigrationMySQL(t *testing.T) {
	db := openTenantModelMigrationTestDB(t)

	if err := db.Exec("DROP TABLE IF EXISTS tenant_model").Error; err != nil {
		t.Fatalf("drop tenant_model test table: %v", err)
	}
	t.Cleanup(func() {
		if err := db.Exec("DROP TABLE IF EXISTS tenant_model").Error; err != nil {
			t.Errorf("cleanup tenant_model test table: %v", err)
		}
	})

	if err := db.Exec(`
		CREATE TABLE tenant_model (
			id VARCHAR(32) NOT NULL PRIMARY KEY,
			model_name VARCHAR(128) NOT NULL,
			provider_id VARCHAR(32) NOT NULL,
			instance_id VARCHAR(32) NOT NULL,
			model_type VARCHAR(64) NULL,
			status VARCHAR(32) NOT NULL,
			extra VARCHAR(1024) NOT NULL
		)
	`).Error; err != nil {
		t.Fatalf("create tenant_model test table: %v", err)
	}

	fixtures := []tenantModelMigrationFixture{
		{ID: "chat-id", ModelName: "chat-model", ProviderID: "provider-a", InstanceID: "instance-a", ModelType: "chat", Status: "active", Extra: `{"slot":"chat"}`},
		{ID: "embedding-id", ModelName: "embedding-model", ProviderID: "provider-a", InstanceID: "instance-a", ModelType: "embedding", Status: "active", Extra: `{"slot":"embedding"}`},
		{ID: "asr-id", ModelName: "asr-model", ProviderID: "provider-a", InstanceID: "instance-a", ModelType: "asr", Status: "disabled", Extra: `{"slot":"asr"}`},
		{ID: "speech-id", ModelName: "speech-model", ProviderID: "provider-b", InstanceID: "instance-b", ModelType: "speech2text", Status: "active", Extra: `{"slot":"speech"}`},
		{ID: "vision-id", ModelName: "vision-model", ProviderID: "provider-b", InstanceID: "instance-b", ModelType: "vision", Status: "active", Extra: `{"slot":"vision"}`},
		{ID: "image-id", ModelName: "image-model", ProviderID: "provider-b", InstanceID: "instance-b", ModelType: "image2text", Status: "active", Extra: `{"slot":"image"}`},
		{ID: "rerank-id", ModelName: "rerank-model", ProviderID: "provider-c", InstanceID: "instance-c", ModelType: "rerank", Status: "active", Extra: `{"slot":"rerank"}`},
		{ID: "tts-id", ModelName: "tts-model", ProviderID: "provider-c", InstanceID: "instance-c", ModelType: "tts", Status: "active", Extra: `{"slot":"tts"}`},
		{ID: "ocr-id", ModelName: "ocr-model", ProviderID: "provider-c", InstanceID: "instance-c", ModelType: "ocr", Status: "active", Extra: `{"slot":"ocr"}`},
		{ID: "combined-id", ModelName: "combined-model", ProviderID: "provider-d", InstanceID: "instance-d", ModelType: "65", Status: "active", Extra: `{"slot":"combined"}`},
		{ID: "duplicate-a", ModelName: "duplicate-model", ProviderID: "provider-e", InstanceID: "instance-e", ModelType: "chat", Status: "active", Extra: `{"slot":"duplicate-a"}`},
		{ID: "duplicate-b", ModelName: "duplicate-model", ProviderID: "provider-e", InstanceID: "instance-e", ModelType: "chat", Status: "inactive", Extra: `{"slot":"duplicate-b"}`},
	}
	for _, fixture := range fixtures {
		if err := db.Exec(
			"INSERT INTO tenant_model (id, model_name, provider_id, instance_id, model_type, status, extra) VALUES (?, ?, ?, ?, ?, ?, ?)",
			fixture.ID,
			fixture.ModelName,
			fixture.ProviderID,
			fixture.InstanceID,
			fixture.ModelType,
			fixture.Status,
			fixture.Extra,
		).Error; err != nil {
			t.Fatalf("insert tenant_model fixture %s: %v", fixture.ID, err)
		}
	}

	var wait sync.WaitGroup
	errors := make(chan error, 2)
	for range 2 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			errors <- RunRequiredStartupMigrations(t.Context(), db, false)
		}()
	}
	wait.Wait()
	close(errors)
	for err := range errors {
		if err != nil {
			t.Fatalf("concurrent startup migration: %v", err)
		}
	}

	assertTenantModelMigrationColumn(t, db)

	wantMasks := map[string]int{
		"chat-id":      1,
		"embedding-id": 2,
		"asr-id":       4,
		"speech-id":    4,
		"vision-id":    8,
		"image-id":     8,
		"rerank-id":    16,
		"tts-id":       32,
		"ocr-id":       64,
		"combined-id":  65,
		"duplicate-a":  1,
		"duplicate-b":  1,
	}
	for _, fixture := range fixtures {
		var got struct {
			ID         string
			ModelName  string
			ProviderID string
			InstanceID string
			ModelType  int
			Status     string
			Extra      string
		}
		if err := db.Table("tenant_model").Where("id = ?", fixture.ID).Take(&got).Error; err != nil {
			t.Fatalf("read migrated fixture %s: %v", fixture.ID, err)
		}
		if got.ModelType != wantMasks[fixture.ID] {
			t.Errorf("fixture %s model_type = %d, want %d", fixture.ID, got.ModelType, wantMasks[fixture.ID])
		}
		if got.ID != fixture.ID || got.ModelName != fixture.ModelName || got.ProviderID != fixture.ProviderID || got.InstanceID != fixture.InstanceID || got.Status != fixture.Status || got.Extra != fixture.Extra {
			t.Errorf("fixture %s metadata changed: %#v", fixture.ID, got)
		}
	}

	if err := RunRequiredStartupMigrations(t.Context(), db, false); err != nil {
		t.Fatalf("idempotent startup migration: %v", err)
	}
	assertTenantModelMigrationColumn(t, db)

	var duplicateCount int64
	if err := db.Raw(`
		SELECT COUNT(*)
		FROM (
			SELECT provider_id, instance_id, model_name
			FROM tenant_model
			GROUP BY provider_id, instance_id, model_name
			HAVING COUNT(*) > 1
		) AS duplicate_logical_models
	`).Scan(&duplicateCount).Error; err != nil {
		t.Fatalf("inspect migrated logical model identities: %v", err)
	}
	if duplicateCount != 1 {
		t.Fatalf("migrated tenant_model has %d duplicate logical model group(s), want 1 preserved", duplicateCount)
	}
}

func TestTenantModelCreateSerializesOnProviderLockMySQL(t *testing.T) {
	db := openTenantModelMigrationTestDB(t)
	for _, table := range []string{"tenant_model", "tenant_model_instance", "tenant_model_provider"} {
		if err := db.Exec("DROP TABLE IF EXISTS " + table).Error; err != nil {
			t.Fatalf("drop %s test table: %v", table, err)
		}
	}
	t.Cleanup(func() {
		for _, table := range []string{"tenant_model", "tenant_model_instance", "tenant_model_provider"} {
			if err := db.Exec("DROP TABLE IF EXISTS " + table).Error; err != nil {
				t.Errorf("cleanup %s test table: %v", table, err)
			}
		}
	})
	if err := db.Exec(`
		CREATE TABLE tenant_model_provider (
			id VARCHAR(32) NOT NULL PRIMARY KEY,
			provider_name VARCHAR(128) NOT NULL,
			tenant_id VARCHAR(32) NOT NULL
		) ENGINE=InnoDB
	`).Error; err != nil {
		t.Fatalf("create tenant_model_provider test table: %v", err)
	}
	if err := db.Exec(`
		CREATE TABLE tenant_model_instance (
			id VARCHAR(32) NOT NULL PRIMARY KEY,
			instance_name VARCHAR(128) NOT NULL,
			provider_id VARCHAR(32) NOT NULL,
			api_key VARCHAR(512) NOT NULL,
			status VARCHAR(32) NOT NULL,
			extra VARCHAR(512) NOT NULL
		) ENGINE=InnoDB
	`).Error; err != nil {
		t.Fatalf("create tenant_model_instance test table: %v", err)
	}
	if err := db.Exec(`
		CREATE TABLE tenant_model (
			id VARCHAR(32) NOT NULL PRIMARY KEY,
			model_name VARCHAR(128) NOT NULL,
			provider_id VARCHAR(32) NOT NULL,
			instance_id VARCHAR(32) NOT NULL,
			model_type INT NOT NULL,
			status VARCHAR(32) NOT NULL,
			extra VARCHAR(1024) NOT NULL
		) ENGINE=InnoDB
	`).Error; err != nil {
		t.Fatalf("create tenant_model test table: %v", err)
	}
	if err := db.Exec(
		"INSERT INTO tenant_model_provider (id, provider_name, tenant_id) VALUES (?, ?, ?)",
		"provider-lock", "OpenAI", "tenant-lock",
	).Error; err != nil {
		t.Fatalf("seed provider: %v", err)
	}
	if err := db.Exec(
		"INSERT INTO tenant_model_instance (id, instance_name, provider_id, api_key, status, extra) VALUES (?, ?, ?, ?, ?, ?)",
		"instance-lock", "default", "provider-lock", "key", "active", "{}",
	).Error; err != nil {
		t.Fatalf("seed instance: %v", err)
	}

	firstLocked := make(chan struct{})
	releaseFirst := make(chan struct{})
	firstDone := make(chan error, 1)
	go func() {
		firstDone <- db.Transaction(func(tx *gorm.DB) error {
			if _, err := NewTenantModelProviderDAO().GetByIDAndTenantIDForUpdate(
				t.Context(), tx, "provider-lock", "tenant-lock",
			); err != nil {
				return err
			}
			close(firstLocked)
			<-releaseFirst
			return NewTenantModelDAO().Create(
				t.Context(), tx,
				&entity.TenantModel{ID: "model-first", ModelName: "same-model", ProviderID: "provider-lock", InstanceID: "instance-lock", ModelType: int(entity.ModelTypeChat), Status: "active", Extra: "{}"},
			)
		})
	}()
	<-firstLocked

	secondDone := make(chan error, 1)
	go func() {
		secondDone <- db.Transaction(func(tx *gorm.DB) error {
			if _, err := NewTenantModelProviderDAO().GetByIDAndTenantIDForUpdate(
				t.Context(), tx, "provider-lock", "tenant-lock",
			); err != nil {
				return err
			}
			_, err := NewTenantModelDAO().GetModelByProviderIDAndInstanceIDAndModelName(
				t.Context(), tx, "provider-lock", "instance-lock", "same-model",
			)
			if err == nil {
				return fmt.Errorf("same-model already exists")
			}
			if !errors.Is(err, gorm.ErrRecordNotFound) {
				return err
			}
			return NewTenantModelDAO().Create(
				t.Context(), tx,
				&entity.TenantModel{ID: "model-second", ModelName: "same-model", ProviderID: "provider-lock", InstanceID: "instance-lock", ModelType: int(entity.ModelTypeChat), Status: "active", Extra: "{}"},
			)
		})
	}()

	select {
	case err := <-secondDone:
		t.Fatalf("second create completed before provider lock released: %v", err)
	case <-time.After(150 * time.Millisecond):
	}
	close(releaseFirst)
	if err := <-firstDone; err != nil {
		t.Fatalf("first create error = %v", err)
	}
	if err := <-secondDone; err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("second create error = %v, want already exists", err)
	}

	var count int64
	if err := db.Model(&entity.TenantModel{}).
		Where("provider_id = ? AND instance_id = ? AND model_name = ?", "provider-lock", "instance-lock", "same-model").
		Count(&count).Error; err != nil {
		t.Fatalf("count same-name models: %v", err)
	}
	if count != 1 {
		t.Fatalf("same-name model count = %d, want 1", count)
	}
}

func TestTenantModelInstanceCreateSerializesOnProviderLockMySQL(t *testing.T) {
	db := openTenantModelMigrationTestDB(t)
	for _, table := range []string{"tenant_model_instance", "tenant_model_provider"} {
		if err := db.Exec("DROP TABLE IF EXISTS " + table).Error; err != nil {
			t.Fatalf("drop %s test table: %v", table, err)
		}
	}
	t.Cleanup(func() {
		for _, table := range []string{"tenant_model_instance", "tenant_model_provider"} {
			if err := db.Exec("DROP TABLE IF EXISTS " + table).Error; err != nil {
				t.Errorf("cleanup %s test table: %v", table, err)
			}
		}
	})
	if err := db.Exec(`
		CREATE TABLE tenant_model_provider (
			id VARCHAR(32) NOT NULL PRIMARY KEY,
			provider_name VARCHAR(128) NOT NULL,
			tenant_id VARCHAR(32) NOT NULL
		) ENGINE=InnoDB
	`).Error; err != nil {
		t.Fatalf("create tenant_model_provider test table: %v", err)
	}
	if err := db.Exec(`
		CREATE TABLE tenant_model_instance (
			id VARCHAR(32) NOT NULL PRIMARY KEY,
			instance_name VARCHAR(128) NOT NULL,
			provider_id VARCHAR(32) NOT NULL,
			api_key VARCHAR(512) NOT NULL,
			status VARCHAR(32) NOT NULL,
			extra VARCHAR(512) NOT NULL
		) ENGINE=InnoDB
	`).Error; err != nil {
		t.Fatalf("create tenant_model_instance test table: %v", err)
	}
	if err := db.Exec(
		"INSERT INTO tenant_model_provider (id, provider_name, tenant_id) VALUES (?, ?, ?)",
		"provider-lock",
		"OpenAI",
		"tenant-lock",
	).Error; err != nil {
		t.Fatalf("seed provider: %v", err)
	}

	firstLocked := make(chan struct{})
	releaseFirst := make(chan struct{})
	firstDone := make(chan error, 1)
	go func() {
		firstDone <- db.Transaction(func(tx *gorm.DB) error {
			if _, err := NewTenantModelProviderDAO().GetByIDAndTenantIDForUpdate(
				t.Context(), tx, "provider-lock", "tenant-lock",
			); err != nil {
				return err
			}
			close(firstLocked)
			<-releaseFirst
			return NewTenantModelInstanceDAO().Create(
				t.Context(),
				tx,
				&entity.TenantModelInstance{ID: "instance-first", InstanceName: "same-name", ProviderID: "provider-lock", APIKey: "key", Status: "active", Extra: "{}"},
			)
		})
	}()
	<-firstLocked

	secondDone := make(chan error, 1)
	go func() {
		secondDone <- db.Transaction(func(tx *gorm.DB) error {
			if _, err := NewTenantModelProviderDAO().GetByIDAndTenantIDForUpdate(
				t.Context(), tx, "provider-lock", "tenant-lock",
			); err != nil {
				return err
			}
			return NewTenantModelInstanceDAO().Create(
				t.Context(),
				tx,
				&entity.TenantModelInstance{ID: "instance-second", InstanceName: "same-name", ProviderID: "provider-lock", APIKey: "key", Status: "active", Extra: "{}"},
			)
		})
	}()

	select {
	case err := <-secondDone:
		t.Fatalf("second create completed before provider lock released: %v", err)
	case <-time.After(150 * time.Millisecond):
	}
	close(releaseFirst)
	if err := <-firstDone; err != nil {
		t.Fatalf("first create error = %v", err)
	}
	if err := <-secondDone; err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("second create error = %v, want already exists", err)
	}

	var count int64
	if err := db.Model(&entity.TenantModelInstance{}).
		Where("provider_id = ? AND instance_name = ?", "provider-lock", "same-name").
		Count(&count).Error; err != nil {
		t.Fatalf("count same-name instances: %v", err)
	}
	if count != 1 {
		t.Fatalf("same-name instance count = %d, want 1", count)
	}
}

func openTenantModelMigrationTestDB(t *testing.T) *gorm.DB {
	t.Helper()

	dsn := os.Getenv(tenantModelMigrationTestDSNEnv)
	if dsn == "" {
		t.Skipf("set %s to a disposable MySQL database ending in _test", tenantModelMigrationTestDSNEnv)
	}
	databaseName, err := mysqlDatabaseNameFromDSN(dsn)
	if err != nil {
		t.Fatalf("parse %s: %v", tenantModelMigrationTestDSNEnv, err)
	}
	if !strings.HasSuffix(databaseName, "_test") {
		t.Fatalf("refusing MySQL database %q: test database name must end in _test", databaseName)
	}

	db, err := gorm.Open(gormmysql.Open(dsn), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Silent),
	})
	if err != nil {
		t.Fatalf("connect MySQL test database: %v", err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("get MySQL test database pool: %v", err)
	}
	t.Cleanup(func() {
		if err := sqlDB.Close(); err != nil {
			t.Errorf("close MySQL test database pool: %v", err)
		}
	})
	return db
}

func mysqlDatabaseNameFromDSN(dsn string) (string, error) {
	config, err := drivermysql.ParseDSN(dsn)
	if err != nil {
		return "", err
	}
	if config.DBName == "" {
		return "", fmt.Errorf("database name is missing")
	}
	return config.DBName, nil
}

func assertTenantModelMigrationColumn(t *testing.T, db *gorm.DB) {
	t.Helper()

	var metadata struct {
		DataType   string `gorm:"column:DATA_TYPE"`
		IsNullable string `gorm:"column:IS_NULLABLE"`
	}
	result := db.Raw(`
		SELECT DATA_TYPE, IS_NULLABLE
		FROM INFORMATION_SCHEMA.COLUMNS
		WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'tenant_model' AND COLUMN_NAME = 'model_type'
	`).Scan(&metadata)
	if result.Error != nil {
		t.Fatalf("inspect migrated tenant_model.model_type: %v", result.Error)
	}
	if result.RowsAffected != 1 || !strings.EqualFold(metadata.DataType, "int") || !strings.EqualFold(metadata.IsNullable, "NO") {
		t.Fatalf("tenant_model.model_type metadata = %#v, rows = %d; want INT NOT NULL", metadata, result.RowsAffected)
	}
}
