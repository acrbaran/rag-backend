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
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

const (
	migrationMockDatabaseName  = "ragflow_migration_test"
	getStartupDatabaseSQL      = "SELECT DATABASE()"
	getStartupMigrationLockSQL = "SELECT GET_LOCK(?, ?) AS lock_result"
	releaseStartupLockSQL      = "SELECT RELEASE_LOCK(?) AS lock_result"
	tenantModelTableSQL        = "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'tenant_model'"
	tenantModelColumnSQL       = "SELECT DATA_TYPE, IS_NULLABLE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'tenant_model' AND COLUMN_NAME = 'model_type'"
)

func newMigrationMockDB(t *testing.T) (*gorm.DB, sqlmock.Sqlmock) {
	t.Helper()

	sqlDB, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock.New: %v", err)
	}
	t.Cleanup(func() {
		mock.ExpectClose()
		if err := sqlDB.Close(); err != nil {
			t.Errorf("close migration mock database: %v", err)
		}
	})

	db, err := gorm.Open(mysql.New(mysql.Config{
		Conn:                      sqlDB,
		SkipInitializeWithVersion: true,
	}), &gorm.Config{
		DisableAutomaticPing: true,
		Logger:               logger.Default.LogMode(logger.Silent),
	})
	if err != nil {
		t.Fatalf("gorm.Open: %v", err)
	}
	return db, mock
}

func expectStartupMigrationLock(mock sqlmock.Sqlmock, acquired int64) {
	mock.ExpectQuery(regexp.QuoteMeta(getStartupDatabaseSQL)).
		WillReturnRows(sqlmock.NewRows([]string{"DATABASE()"}).AddRow(migrationMockDatabaseName))
	mock.ExpectQuery(regexp.QuoteMeta(getStartupMigrationLockSQL)).
		WithArgs(
			requiredStartupMigrationLockName(migrationMockDatabaseName),
			int(requiredStartupMigrationLockTimeout/time.Second),
		).
		WillReturnRows(sqlmock.NewRows([]string{"lock_result"}).AddRow(acquired))
}

func expectStartupMigrationUnlock(mock sqlmock.Sqlmock, released int64) {
	mock.ExpectQuery(regexp.QuoteMeta(releaseStartupLockSQL)).
		WithArgs(requiredStartupMigrationLockName(migrationMockDatabaseName)).
		WillReturnRows(sqlmock.NewRows([]string{"lock_result"}).AddRow(released))
}

func expectTenantModelTable(mock sqlmock.Sqlmock, count int64) {
	mock.ExpectQuery(regexp.QuoteMeta(tenantModelTableSQL)).
		WillReturnRows(sqlmock.NewRows([]string{"COUNT(*)"}).AddRow(count))
}

func expectTenantModelColumn(mock sqlmock.Sqlmock, dataType, nullable string) {
	mock.ExpectQuery(regexp.QuoteMeta(tenantModelColumnSQL)).
		WillReturnRows(sqlmock.NewRows([]string{"DATA_TYPE", "IS_NULLABLE"}).
			AddRow(dataType, nullable))
}

func TestRequiredStartupMigrationLockNameScopesDatabase(t *testing.T) {
	first := requiredStartupMigrationLockName("ragflow_one")
	second := requiredStartupMigrationLockName("ragflow_two")

	if first == second {
		t.Fatalf("required startup migration lock names match: %q", first)
	}
	if first != requiredStartupMigrationLockName("ragflow_one") {
		t.Fatalf("required startup migration lock name is not stable: %q", first)
	}
	if len(first) > 64 || len(second) > 64 {
		t.Fatalf("required startup migration lock name exceeds MySQL limit: %q, %q", first, second)
	}
}

func TestRunRequiredStartupMigrationsCreatesMissingTenantModelTableInMigrationMode(t *testing.T) {
	db, mock := newMigrationMockDB(t)
	expectStartupMigrationLock(mock, 1)
	expectTenantModelTable(mock, 0)
	mock.ExpectExec("CREATE TABLE.*tenant_model").
		WillReturnResult(sqlmock.NewResult(0, 0))
	expectTenantModelTable(mock, 1)
	expectTenantModelColumn(mock, "int", "NO")
	mock.ExpectQuery("SELECT COUNT\\(\\*\\).*FROM tenant_model.*").
		WillReturnRows(sqlmock.NewRows([]string{"COUNT(*)"}).AddRow(0))
	expectStartupMigrationUnlock(mock, 1)

	if err := RunRequiredStartupMigrations(t.Context(), db, true); err != nil {
		t.Fatalf("RunRequiredStartupMigrations() error = %v, want nil", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("startup migration SQL contract: %v", err)
	}
}

func TestRunRequiredStartupMigrationsReleasesLockAfterMigrationError(t *testing.T) {
	db, mock := newMigrationMockDB(t)
	expectStartupMigrationLock(mock, 1)
	migrationErr := errors.New("metadata unavailable")
	mock.ExpectQuery(regexp.QuoteMeta(tenantModelTableSQL)).WillReturnError(migrationErr)
	expectStartupMigrationUnlock(mock, 1)

	err := RunRequiredStartupMigrations(t.Context(), db, false)
	if !errors.Is(err, migrationErr) {
		t.Fatalf("RunRequiredStartupMigrations() error = %v, want wrapped migration error", err)
	}
	if err = mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("startup migration SQL contract: %v", err)
	}
}

func TestRunRequiredStartupMigrationsFailsWhenLockIsUnavailable(t *testing.T) {
	db, mock := newMigrationMockDB(t)
	expectStartupMigrationLock(mock, 0)

	err := RunRequiredStartupMigrations(t.Context(), db, false)
	if err == nil || !strings.Contains(err.Error(), "advisory lock") {
		t.Fatalf("RunRequiredStartupMigrations() error = %v, want advisory lock error", err)
	}
	if err = mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("startup migration SQL contract: %v", err)
	}
}

func TestRunRequiredStartupMigrationsFailsWhenUnlockFails(t *testing.T) {
	db, mock := newMigrationMockDB(t)
	expectStartupMigrationLock(mock, 1)
	expectTenantModelTable(mock, 1)
	expectTenantModelColumn(mock, "int", "NO")
	mock.ExpectQuery("SELECT COUNT\\(\\*\\).*FROM tenant_model.*").
		WillReturnRows(sqlmock.NewRows([]string{"COUNT(*)"}).AddRow(0))
	expectStartupMigrationUnlock(mock, 0)

	err := RunRequiredStartupMigrations(t.Context(), db, false)
	if err == nil || !strings.Contains(err.Error(), "release advisory lock") {
		t.Fatalf("RunRequiredStartupMigrations() error = %v, want advisory unlock error", err)
	}
	if err = mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("startup migration SQL contract: %v", err)
	}
}

func TestRunRequiredStartupMigrationsReleasesLockAfterContextCancellation(t *testing.T) {
	db, mock := newMigrationMockDB(t)
	ctx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()
	expectStartupMigrationLock(mock, 1)
	mock.ExpectQuery(regexp.QuoteMeta(tenantModelTableSQL)).
		WillDelayFor(200 * time.Millisecond).
		WillReturnRows(sqlmock.NewRows([]string{"COUNT(*)"}).AddRow(0))
	expectStartupMigrationUnlock(mock, 1)

	err := RunRequiredStartupMigrations(ctx, db, false)
	if err == nil {
		t.Fatal("RunRequiredStartupMigrations() error = nil, want canceled migration error")
	}
	if strings.Contains(err.Error(), "release advisory lock") {
		t.Fatalf("RunRequiredStartupMigrations() error = %v, lock release inherited canceled context", err)
	}
	if err = mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("startup migration must release lock after context cancellation: %v", err)
	}
}

func TestMigrateTenantModelTypeSchemaRejectsMissingTable(t *testing.T) {
	db, mock := newMigrationMockDB(t)
	expectTenantModelTable(mock, 0)

	err := migrateTenantModelTypeSchema(t.Context(), db)
	if err == nil || !strings.Contains(err.Error(), "tenant_model table is missing") {
		t.Fatalf("migrateTenantModelTypeSchema() error = %v, want missing table error", err)
	}
	if err = mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("tenant model migration SQL contract: %v", err)
	}
}

func TestMigrateTenantModelTypeSchemaRejectsMissingColumn(t *testing.T) {
	db, mock := newMigrationMockDB(t)
	expectTenantModelTable(mock, 1)
	mock.ExpectQuery(regexp.QuoteMeta(tenantModelColumnSQL)).
		WillReturnRows(sqlmock.NewRows([]string{"DATA_TYPE", "IS_NULLABLE"}))

	err := migrateTenantModelTypeSchema(t.Context(), db)
	if err == nil || !strings.Contains(err.Error(), "model_type column is missing") {
		t.Fatalf("migrateTenantModelTypeSchema() error = %v, want missing column error", err)
	}
	if err = mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("tenant model migration SQL contract: %v", err)
	}
}

func TestMigrateTenantModelTypeSchemaConvergedColumnIsNoOp(t *testing.T) {
	db, mock := newMigrationMockDB(t)
	expectTenantModelTable(mock, 1)
	expectTenantModelColumn(mock, "int", "NO")
	mock.ExpectQuery("SELECT COUNT\\(\\*\\).*FROM tenant_model.*").
		WillReturnRows(sqlmock.NewRows([]string{"COUNT(*)"}).AddRow(0))

	if err := migrateTenantModelTypeSchema(t.Context(), db); err != nil {
		t.Fatalf("migrateTenantModelTypeSchema() error = %v, want nil", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("tenant model migration SQL contract: %v", err)
	}
}

func TestMigrateTenantModelTypeSchemaRejectsUnsafeValuesInConvergedColumn(t *testing.T) {
	db, mock := newMigrationMockDB(t)
	expectTenantModelTable(mock, 1)
	expectTenantModelColumn(mock, "int", "NO")
	mock.ExpectQuery("SELECT COUNT\\(\\*\\).*FROM tenant_model.*").
		WillReturnRows(sqlmock.NewRows([]string{"COUNT(*)"}).AddRow(1))

	err := migrateTenantModelTypeSchema(t.Context(), db)
	if err == nil || !strings.Contains(err.Error(), "1 unsupported") {
		t.Fatalf("migrateTenantModelTypeSchema() error = %v, want unsupported value count", err)
	}
	if err = mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("tenant model migration SQL contract: %v", err)
	}
}

func TestMigrateTenantModelTypeSchemaRejectsUnsupportedSourceTypeBeforeValueScan(t *testing.T) {
	db, mock := newMigrationMockDB(t)
	expectTenantModelTable(mock, 1)
	expectTenantModelColumn(mock, "bigint", "NO")

	err := migrateTenantModelTypeSchema(t.Context(), db)
	if err == nil || !strings.Contains(err.Error(), `unsupported source type "bigint"`) {
		t.Fatalf("migrateTenantModelTypeSchema() error = %v, want unsupported source type", err)
	}
	if err = mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("tenant model migration SQL contract: %v", err)
	}
}

func TestMigrateTenantModelTypeSchemaRepairsNullableIntegerColumn(t *testing.T) {
	db, mock := newMigrationMockDB(t)
	expectTenantModelTable(mock, 1)
	expectTenantModelColumn(mock, "int", "YES")
	mock.ExpectQuery("SELECT COUNT\\(\\*\\).*FROM tenant_model.*").
		WillReturnRows(sqlmock.NewRows([]string{"COUNT(*)"}).AddRow(0))
	mock.ExpectExec(regexp.QuoteMeta("ALTER TABLE tenant_model MODIFY COLUMN model_type INT NOT NULL")).
		WillReturnResult(sqlmock.NewResult(0, 0))
	expectTenantModelColumn(mock, "int", "NO")
	mock.ExpectQuery("SELECT COUNT\\(\\*\\).*FROM tenant_model.*").
		WillReturnRows(sqlmock.NewRows([]string{"COUNT(*)"}).AddRow(0))

	if err := migrateTenantModelTypeSchema(t.Context(), db); err != nil {
		t.Fatalf("migrateTenantModelTypeSchema() error = %v, want nil", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("tenant model migration SQL contract: %v", err)
	}
}

func TestMigrateTenantModelTypeSchemaRejectsUnsafeValuesBeforeMutation(t *testing.T) {
	db, mock := newMigrationMockDB(t)
	expectTenantModelTable(mock, 1)
	expectTenantModelColumn(mock, "varchar", "YES")
	mock.ExpectQuery("SELECT COUNT\\(\\*\\).*FROM tenant_model.*").
		WillReturnRows(sqlmock.NewRows([]string{"COUNT(*)"}).AddRow(2))

	err := migrateTenantModelTypeSchema(t.Context(), db)
	if err == nil || !strings.Contains(err.Error(), "2 unsupported") {
		t.Fatalf("migrateTenantModelTypeSchema() error = %v, want unsupported value count", err)
	}
	if err = mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("tenant model migration mutated unsafe data: %v", err)
	}
}
