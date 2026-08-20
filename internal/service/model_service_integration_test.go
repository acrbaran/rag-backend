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

package service

import (
	"context"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	drivermysql "github.com/go-sql-driver/mysql"
	gormmysql "gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"ragflow/internal/common"
	"ragflow/internal/dao"
	"ragflow/internal/entity"
)

const modelServiceMySQLTestDSNEnv = "RAGFLOW_MYSQL_TEST_DSN"

func TestAddModelSerializesDuplicateCreationOnProviderLockMySQL(t *testing.T) {
	db := openModelServiceMySQLTestDB(t)
	originalDB := dao.DB
	dao.DB = db
	t.Cleanup(func() { dao.DB = originalDB })

	for _, table := range []string{
		"tenant_model",
		"tenant_model_instance",
		"tenant_model_provider",
		"user_tenant",
	} {
		if err := db.Exec("DROP TABLE IF EXISTS " + table).Error; err != nil {
			t.Fatalf("drop %s test table: %v", table, err)
		}
	}
	t.Cleanup(func() {
		for _, table := range []string{
			"tenant_model",
			"tenant_model_instance",
			"tenant_model_provider",
			"user_tenant",
		} {
			if err := db.Exec("DROP TABLE IF EXISTS " + table).Error; err != nil {
				t.Errorf("cleanup %s test table: %v", table, err)
			}
		}
	})

	statements := []string{
		`CREATE TABLE user_tenant (
			id VARCHAR(32) NOT NULL PRIMARY KEY,
			user_id VARCHAR(32) NOT NULL,
			tenant_id VARCHAR(32) NOT NULL,
			role VARCHAR(32) NOT NULL,
			invited_by VARCHAR(32) NOT NULL,
			status VARCHAR(1) NOT NULL,
			create_time BIGINT NULL,
			create_date DATETIME NULL,
			update_time BIGINT NULL,
			update_date DATETIME NULL
		) ENGINE=InnoDB`,
		`CREATE TABLE tenant_model_provider (
			id VARCHAR(32) NOT NULL PRIMARY KEY,
			provider_name VARCHAR(128) NOT NULL,
			tenant_id VARCHAR(32) NOT NULL,
			create_time BIGINT NULL,
			create_date DATETIME NULL,
			update_time BIGINT NULL,
			update_date DATETIME NULL
		) ENGINE=InnoDB`,
		`CREATE TABLE tenant_model_instance (
			id VARCHAR(32) NOT NULL PRIMARY KEY,
			instance_name VARCHAR(128) NOT NULL,
			provider_id VARCHAR(32) NOT NULL,
			api_key VARCHAR(512) NOT NULL,
			status VARCHAR(32) NOT NULL,
			extra VARCHAR(512) NOT NULL,
			create_time BIGINT NULL,
			create_date DATETIME NULL,
			update_time BIGINT NULL,
			update_date DATETIME NULL
		) ENGINE=InnoDB`,
		`CREATE TABLE tenant_model (
			id VARCHAR(32) NOT NULL PRIMARY KEY,
			model_name VARCHAR(128) NOT NULL,
			provider_id VARCHAR(32) NOT NULL,
			instance_id VARCHAR(32) NOT NULL,
			model_type INT NOT NULL,
			status VARCHAR(32) NOT NULL,
			extra VARCHAR(1024) NOT NULL,
			create_time BIGINT NULL,
			create_date DATETIME NULL,
			update_time BIGINT NULL,
			update_date DATETIME NULL
		) ENGINE=InnoDB`,
	}
	for _, statement := range statements {
		if err := db.Exec(statement).Error; err != nil {
			t.Fatalf("create model service test table: %v", err)
		}
	}

	seedStatements := []struct {
		query string
		args  []interface{}
	}{
		{
			query: "INSERT INTO user_tenant (id, user_id, tenant_id, role, invited_by, status) VALUES (?, ?, ?, ?, ?, ?)",
			args:  []interface{}{"membership-lock", "user-lock", "tenant-lock", "owner", "user-lock", "1"},
		},
		{
			query: "INSERT INTO tenant_model_provider (id, provider_name, tenant_id) VALUES (?, ?, ?)",
			args:  []interface{}{"provider-lock", "OpenAI", "tenant-lock"},
		},
		{
			query: "INSERT INTO tenant_model_instance (id, instance_name, provider_id, api_key, status, extra) VALUES (?, ?, ?, ?, ?, ?)",
			args:  []interface{}{"instance-lock", "default", "provider-lock", "test-key", "active", "{}"},
		},
	}
	for _, seed := range seedStatements {
		if err := db.Exec(seed.query, seed.args...).Error; err != nil {
			t.Fatalf("seed model service test row: %v", err)
		}
	}

	blocker := db.Begin()
	if blocker.Error != nil {
		t.Fatalf("begin provider blocker transaction: %v", blocker.Error)
	}
	blockerActive := true
	t.Cleanup(func() {
		if blockerActive {
			if err := blocker.Rollback().Error; err != nil {
				t.Errorf("cleanup provider blocker transaction: %v", err)
			}
		}
	})
	if err := blocker.Exec(
		"SELECT id FROM tenant_model_provider WHERE id = ? FOR UPDATE",
		"provider-lock",
	).Error; err != nil {
		t.Fatalf("lock provider row: %v", err)
	}

	lockAttempts := make(chan struct{}, 2)
	const callbackName = "observe_add_model_provider_lock"
	if err := db.Callback().Query().Before("gorm:query").Register(
		callbackName,
		func(tx *gorm.DB) {
			if tx.Statement.Table != "tenant_model_provider" {
				return
			}
			if _, locking := tx.Statement.Clauses["FOR"]; !locking {
				return
			}
			select {
			case lockAttempts <- struct{}{}:
			default:
			}
		},
	); err != nil {
		t.Fatalf("register provider lock observer: %v", err)
	}
	t.Cleanup(func() {
		if err := db.Callback().Query().Remove(callbackName); err != nil {
			t.Errorf("remove provider lock observer: %v", err)
		}
	})

	type addModelResult struct {
		code common.ErrorCode
		err  error
	}
	workerCtx, cancelWorkers := context.WithCancel(t.Context())
	defer cancelWorkers()
	results := make(chan addModelResult, 2)
	var workers sync.WaitGroup
	service := NewModelProviderService()
	request := &AddModelRequest{
		ProviderName: "provider-lock",
		InstanceName: "instance-lock",
		ModelName:    "same-model",
		ModelTypes:   []string{"chat"},
	}
	for range 2 {
		workers.Add(1)
		go func() {
			defer workers.Done()
			code, err := service.AddModel(workerCtx, request, "user-lock")
			results <- addModelResult{code: code, err: err}
		}()
	}

	waitForWorkers := func() bool {
		done := make(chan struct{})
		go func() {
			workers.Wait()
			close(done)
		}()
		select {
		case <-done:
			return true
		case <-time.After(5 * time.Second):
			return false
		}
	}

	for attempt := 0; attempt < 2; attempt++ {
		select {
		case <-lockAttempts:
		case <-time.After(5 * time.Second):
			if err := blocker.Rollback().Error; err != nil {
				t.Errorf("release provider blocker after timeout: %v", err)
			}
			blockerActive = false
			cancelWorkers()
			if !waitForWorkers() {
				t.Error("concurrent AddModel calls did not stop after cancellation")
			}
			t.Fatal("concurrent AddModel calls did not reach provider row lock")
		}
	}
	if err := blocker.Rollback().Error; err != nil {
		cancelWorkers()
		if !waitForWorkers() {
			t.Error("concurrent AddModel calls did not stop after blocker rollback failure")
		}
		t.Fatalf("release provider blocker transaction: %v", err)
	}
	blockerActive = false

	outcomes := make([]addModelResult, 0, 2)
	for len(outcomes) < 2 {
		select {
		case outcome := <-results:
			outcomes = append(outcomes, outcome)
		case <-time.After(5 * time.Second):
			cancelWorkers()
			if !waitForWorkers() {
				t.Error("concurrent AddModel calls did not stop after result timeout")
			}
			t.Fatal("timed out waiting for concurrent AddModel results")
		}
	}
	if !waitForWorkers() {
		t.Fatal("concurrent AddModel workers did not exit")
	}
	successes := 0
	conflicts := 0
	for _, outcome := range outcomes {
		switch {
		case outcome.err == nil && outcome.code == common.CodeSuccess:
			successes++
		case outcome.code == common.CodeConflict &&
			errors.Is(outcome.err, errDuplicateExistingModel):
			conflicts++
		default:
			t.Fatalf("AddModel outcome = code %v, error %v", outcome.code, outcome.err)
		}
	}
	if successes != 1 || conflicts != 1 {
		t.Fatalf(
			"AddModel outcomes = %d success, %d conflict; want 1, 1",
			successes,
			conflicts,
		)
	}

	var count int64
	if err := db.Model(&entity.TenantModel{}).
		Where(
			"provider_id = ? AND instance_id = ? AND model_name = ?",
			"provider-lock",
			"instance-lock",
			"same-model",
		).
		Count(&count).Error; err != nil {
		t.Fatalf("count same-name models: %v", err)
	}
	if count != 1 {
		t.Fatalf("same-name model count = %d, want 1", count)
	}
}

func openModelServiceMySQLTestDB(t *testing.T) *gorm.DB {
	t.Helper()

	dsn := os.Getenv(modelServiceMySQLTestDSNEnv)
	if dsn == "" {
		t.Skipf(
			"set %s to a disposable MySQL database ending in _test",
			modelServiceMySQLTestDSNEnv,
		)
	}
	config, err := drivermysql.ParseDSN(dsn)
	if err != nil {
		t.Fatalf("parse %s: invalid DSN", modelServiceMySQLTestDSNEnv)
	}
	if !strings.HasSuffix(config.DBName, "_test") {
		t.Fatalf(
			"refusing MySQL database %q: test database name must end in _test",
			config.DBName,
		)
	}

	db, err := gorm.Open(gormmysql.Open(dsn), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Silent),
	})
	if err != nil {
		t.Fatal("connect disposable MySQL test database failed")
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatal("get MySQL test database pool failed")
	}
	t.Cleanup(func() {
		if err := sqlDB.Close(); err != nil {
			t.Errorf("close MySQL test database pool: %v", err)
		}
	})
	return db
}
