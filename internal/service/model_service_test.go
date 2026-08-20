package service

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"

	"ragflow/internal/common"
	"ragflow/internal/dao"
	"ragflow/internal/entity"
	modelModule "ragflow/internal/entity/models"
	"ragflow/internal/utility"
)

type remoteModelProbeDriver struct {
	*modelModule.DummyModel
	newInstanceResult   modelModule.ModelDriver
	remoteModels        []modelModule.ListModelResponse
	listErr             error
	connectionErr       error
	listCalls           int
	embedCalls          int
	connectionCalls     int
	connectionAPIConfig modelModule.APIConfig
}

func (d *remoteModelProbeDriver) NewInstance(
	map[string]string,
) modelModule.ModelDriver {
	return d.newInstanceResult
}

func (d *remoteModelProbeDriver) ListModels(context.Context, *modelModule.APIConfig) ([]modelModule.ListModelResponse, error) {
	d.listCalls++
	return d.remoteModels, d.listErr
}

func (d *remoteModelProbeDriver) Embed(context.Context, *string, modelModule.EmbedRequest, *modelModule.APIConfig, *modelModule.EmbeddingConfig, *common.ModelUsage) ([]modelModule.EmbeddingData, error) {
	d.embedCalls++
	return nil, nil
}

func (d *remoteModelProbeDriver) CheckConnection(_ context.Context, apiConfig *modelModule.APIConfig) error {
	d.connectionCalls++
	if apiConfig != nil {
		d.connectionAPIConfig = *apiConfig
	}
	return d.connectionErr
}

func installModelServiceProbeDriver(
	t *testing.T,
	providerName string,
	probe modelModule.ModelDriver,
) {
	t.Helper()
	providerInfo := dao.GetModelProviderManager().FindProvider(providerName)
	if providerInfo == nil {
		t.Fatalf("%s provider metadata missing", providerName)
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })
}

func TestNormalizeModelNameAgainstRemoteCatalogRejectsMissingScope(t *testing.T) {
	service := NewModelProviderService()
	provider := &entity.TenantModelProvider{ProviderName: "OpenAI"}
	instance := &entity.TenantModelInstance{Extra: "{}"}
	tests := []struct {
		name     string
		provider *entity.TenantModelProvider
		instance *entity.TenantModelInstance
		wantErr  string
	}{
		{
			name:     "missing provider",
			instance: instance,
			wantErr:  "provider is required",
		},
		{
			name:     "missing instance",
			provider: provider,
			wantErr:  "instance is required",
		},
		{
			name:     "missing provider driver",
			provider: &entity.TenantModelProvider{ProviderName: "missing-provider"},
			instance: instance,
			wantErr:  "driver not found",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, changed, err := service.normalizeModelNameAgainstRemoteCatalog(
				t.Context(),
				test.provider,
				test.instance,
				"remote-model@openai",
			)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf(
					"normalizeModelNameAgainstRemoteCatalog() error = %v, want %q",
					err,
					test.wantErr,
				)
			}
			if got != "remote-model@openai" || changed {
				t.Fatalf("result = (%q, %v), want unchanged model", got, changed)
			}
		})
	}
}

func TestNormalizeModelNameAgainstRemoteCatalogRejectsUnsupportedBaseURL(t *testing.T) {
	service := NewModelProviderService()
	provider := &entity.TenantModelProvider{ProviderName: "OpenAI"}
	instance := &entity.TenantModelInstance{
		Extra: `{"base_url":"https://models.invalid"}`,
	}
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
	}
	providerInfo := dao.GetModelProviderManager().FindProvider(provider.ProviderName)
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	got, changed, err := service.normalizeModelNameAgainstRemoteCatalog(
		t.Context(),
		provider,
		instance,
		"remote-model@openai",
	)
	if err == nil || !strings.Contains(err.Error(), "does not support custom base_url") {
		t.Fatalf(
			"normalizeModelNameAgainstRemoteCatalog() error = %v, want base URL error",
			err,
		)
	}
	if got != "remote-model@openai" || changed {
		t.Fatalf("result = (%q, %v), want unchanged model", got, changed)
	}
}

func TestNormalizeModelNameAgainstRemoteCatalogRejectsInvalidMetadata(t *testing.T) {
	service := NewModelProviderService()
	provider := &entity.TenantModelProvider{ProviderName: "OpenAI"}
	instance := &entity.TenantModelInstance{Extra: "{"}

	got, changed, err := service.normalizeModelNameAgainstRemoteCatalog(
		t.Context(),
		provider,
		instance,
		"remote-model@openai",
	)
	if err == nil || !strings.Contains(err.Error(), "invalid instance metadata") {
		t.Fatalf("normalizeModelNameAgainstRemoteCatalog() error = %v, want invalid metadata error", err)
	}
	if got != "remote-model@openai" || changed {
		t.Fatalf("result = (%q, %v), want unchanged model", got, changed)
	}
}

func TestNormalizeModelNameAgainstRemoteCatalogRejectsListFailure(t *testing.T) {
	service := NewModelProviderService()
	provider := &entity.TenantModelProvider{ProviderName: "OpenAI"}
	instance := &entity.TenantModelInstance{Extra: "{}"}
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		listErr:    errors.New("catalog unavailable"),
	}
	probe.newInstanceResult = probe
	providerInfo := dao.GetModelProviderManager().FindProvider(provider.ProviderName)
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	got, changed, err := service.normalizeModelNameAgainstRemoteCatalog(
		t.Context(),
		provider,
		instance,
		"remote-model@openai",
	)
	if err == nil || !strings.Contains(err.Error(), "catalog unavailable") {
		t.Fatalf("normalizeModelNameAgainstRemoteCatalog() error = %v, want catalog error", err)
	}
	if got != "remote-model@openai" || changed {
		t.Fatalf("result = (%q, %v), want unchanged model", got, changed)
	}
}

func TestNormalizeModelNameAgainstRemoteCatalogRejectsEmptyCatalog(t *testing.T) {
	service := NewModelProviderService()
	provider := &entity.TenantModelProvider{ProviderName: "OpenAI"}
	instance := &entity.TenantModelInstance{Extra: "{}"}
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		remoteModels: []modelModule.ListModelResponse{
			{Name: "  "},
		},
	}
	probe.newInstanceResult = probe
	providerInfo := dao.GetModelProviderManager().FindProvider(provider.ProviderName)
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	got, changed, err := service.normalizeModelNameAgainstRemoteCatalog(
		t.Context(),
		provider,
		instance,
		"remote-model@openai",
	)
	if err == nil || !strings.Contains(err.Error(), "no usable models") {
		t.Fatalf("normalizeModelNameAgainstRemoteCatalog() error = %v, want empty catalog error", err)
	}
	if got != "remote-model@openai" || changed {
		t.Fatalf("result = (%q, %v), want unchanged model", got, changed)
	}
}

func TestNormalizeModelNameAgainstRemoteCatalogRejectsUnlistedModel(t *testing.T) {
	service := NewModelProviderService()
	provider := &entity.TenantModelProvider{ProviderName: "OpenAI"}
	instance := &entity.TenantModelInstance{Extra: "{}"}
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		remoteModels: []modelModule.ListModelResponse{{
			Name: "different-model",
		}},
	}
	probe.newInstanceResult = probe
	providerInfo := dao.GetModelProviderManager().FindProvider(provider.ProviderName)
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	got, changed, err := service.normalizeModelNameAgainstRemoteCatalog(
		t.Context(),
		provider,
		instance,
		"remote-model@openai",
	)
	if err == nil ||
		!errors.Is(err, errRemoteModelNotFound) ||
		!strings.Contains(err.Error(), "not found in remote model catalog") {
		t.Fatalf("normalizeModelNameAgainstRemoteCatalog() error = %v, want unlisted model error", err)
	}
	if got != "remote-model@openai" || changed {
		t.Fatalf("result = (%q, %v), want unchanged model", got, changed)
	}
}

func TestAddModelClassifiesUnlistedRemoteModelAsDataError(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		remoteModels: []modelModule.ListModelResponse{{
			Name: "different-model",
		}},
	}
	probe.newInstanceResult = probe
	providerInfo := dao.GetModelProviderManager().FindProvider("OpenAI")
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	code, err := NewModelProviderService().AddModel(
		t.Context(),
		&AddModelRequest{
			ProviderName: "OpenAI",
			InstanceName: "default",
			ModelName:    "remote-model@openai",
			ModelTypes:   []string{"embedding"},
		},
		"user-1",
	)
	if err == nil || !strings.Contains(err.Error(), "not found in remote model catalog") {
		t.Fatalf("AddModel() error = %v, want unlisted model error", err)
	}
	if code != common.CodeDataError {
		t.Fatalf("code = %v, want %v", code, common.CodeDataError)
	}
}

func TestNormalizeCreateInstanceModelsUsesOneCatalogSnapshot(t *testing.T) {
	service := NewModelProviderService()
	provider := &entity.TenantModelProvider{ProviderName: "OpenAI"}
	instance := &entity.TenantModelInstance{Extra: "{}"}
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		remoteModels: []modelModule.ListModelResponse{
			{Name: "model-a"},
			{Name: "model-b"},
		},
	}
	probe.newInstanceResult = probe
	providerInfo := dao.GetModelProviderManager().FindProvider(provider.ProviderName)
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	normalized, err := service.normalizeCreateInstanceModels(
		t.Context(),
		provider,
		instance,
		[]CreateInstanceModelInfo{
			{ModelName: "model-a@openai"},
			{ModelName: "model-b@openai"},
		},
	)
	if err != nil {
		t.Fatalf("normalizeCreateInstanceModels() error = %v", err)
	}
	if probe.listCalls != 1 {
		t.Fatalf("ListModels calls = %d, want 1", probe.listCalls)
	}
	if len(normalized) != 2 ||
		normalized[0].ModelName != "model-a" ||
		normalized[1].ModelName != "model-b" {
		t.Fatalf("normalized models = %#v", normalized)
	}
}

func TestAddModelPreservesNonLegacyAtSignWithoutCatalogLookup(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		listErr:    errors.New("catalog must not be called"),
	}
	installModelServiceProbeDriver(t, "OpenAI", probe)

	code, err := NewModelProviderService().AddModel(
		t.Context(),
		&AddModelRequest{
			ProviderName: "OpenAI",
			InstanceName: "default",
			ModelName:    "team@model",
			ModelTypes:   []string{"embedding"},
		},
		"user-1",
	)
	if err != nil {
		t.Fatalf("AddModel() error = %v", err)
	}
	if code != common.CodeSuccess {
		t.Fatalf("code = %v, want %v", code, common.CodeSuccess)
	}
	if probe.listCalls != 0 {
		t.Fatalf("ListModels calls = %d, want 0", probe.listCalls)
	}

	var created entity.TenantModel
	if err := db.Where(
		"provider_id = ? AND instance_id = ? AND model_name = ?",
		"provider-1",
		"instance-1",
		"team@model",
	).Take(&created).Error; err != nil {
		t.Fatalf("reload created model: %v", err)
	}
}

func TestAddModelToInstanceRejectsRemoteCatalogFailure(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		listErr:    errors.New("catalog unavailable"),
	}
	probe.newInstanceResult = probe
	providerInfo := dao.GetModelProviderManager().FindProvider("OpenAI")
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	err := NewModelProviderService().addModelToInstance(
		t.Context(),
		"tenant-1",
		"OpenAI",
		"default",
		CreateInstanceModelInfo{
			ModelName:  "remote-model@openai",
			ModelTypes: []string{"embedding"},
		},
	)
	if err == nil || !strings.Contains(err.Error(), "catalog unavailable") {
		t.Fatalf("addModelToInstance() error = %v, want catalog error", err)
	}

	var count int64
	if err := db.Model(&entity.TenantModel{}).
		Where("model_name = ?", "remote-model@openai").
		Count(&count).Error; err != nil {
		t.Fatalf("failed to count tenant models: %v", err)
	}
	if count != 0 {
		t.Fatalf("created model count = %d, want 0", count)
	}
}

func TestAddModelChecksDuplicateAndCreatesInsideTransaction(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)

	duplicateLookupSeen := false
	duplicateLookupInTransaction := false
	if err := db.Callback().Query().After("gorm:query").Register(
		"observe_add_model_duplicate_lookup",
		func(tx *gorm.DB) {
			if tx.Statement.Table != "tenant_model" ||
				!statementHasStringVariable(tx, "atomic-model") {
				return
			}
			duplicateLookupSeen = true
			_, duplicateLookupInTransaction = tx.Statement.ConnPool.(*sql.Tx)
		},
	); err != nil {
		t.Fatalf("register duplicate lookup observer: %v", err)
	}

	createSeen := false
	createInTransaction := false
	if err := db.Callback().Create().Before("gorm:create").Register(
		"observe_add_model_create",
		func(tx *gorm.DB) {
			model, ok := tx.Statement.Dest.(*entity.TenantModel)
			if !ok || model.ModelName != "atomic-model" {
				return
			}
			createSeen = true
			_, createInTransaction = tx.Statement.ConnPool.(*sql.Tx)
		},
	); err != nil {
		t.Fatalf("register create observer: %v", err)
	}

	code, err := NewModelProviderService().AddModel(
		t.Context(),
		&AddModelRequest{
			ProviderName: "OpenAI",
			InstanceName: "default",
			ModelName:    "atomic-model",
			ModelTypes:   []string{"chat"},
		},
		"user-1",
	)
	if err != nil {
		t.Fatalf("AddModel() error = %v", err)
	}
	if code != common.CodeSuccess {
		t.Fatalf("code = %v, want %v", code, common.CodeSuccess)
	}
	if !duplicateLookupSeen || !duplicateLookupInTransaction {
		t.Fatalf(
			"duplicate lookup = (seen=%v, transaction=%v), want true, true",
			duplicateLookupSeen,
			duplicateLookupInTransaction,
		)
	}
	if !createSeen || !createInTransaction {
		t.Fatalf(
			"create = (seen=%v, transaction=%v), want true, true",
			createSeen,
			createInTransaction,
		)
	}
}

func TestAddModelRejectsExistingModelAsConflict(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)

	code, err := NewModelProviderService().AddModel(
		t.Context(),
		&AddModelRequest{
			ProviderName: "OpenAI",
			InstanceName: "default",
			ModelName:    "gpt-test",
			ModelTypes:   []string{"chat"},
		},
		"user-1",
	)
	if err == nil || !errors.Is(err, errDuplicateExistingModel) {
		t.Fatalf("AddModel() error = %v, want duplicate existing model error", err)
	}
	if code != common.CodeConflict {
		t.Fatalf("code = %v, want %v", code, common.CodeConflict)
	}

	var count int64
	if err := db.Model(&entity.TenantModel{}).
		Where(
			"provider_id = ? AND instance_id = ? AND model_name = ?",
			"provider-1",
			"instance-1",
			"gpt-test",
		).
		Count(&count).Error; err != nil {
		t.Fatalf("count existing model rows: %v", err)
	}
	if count != 1 {
		t.Fatalf("existing model count = %d, want 1", count)
	}
}

func statementHasStringVariable(tx *gorm.DB, want string) bool {
	for _, variable := range tx.Statement.Vars {
		if got, ok := variable.(string); ok && got == want {
			return true
		}
	}
	return false
}

func TestAddModelRejectsRemoteCatalogFailure(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		listErr:    errors.New("catalog unavailable"),
	}
	probe.newInstanceResult = probe
	providerInfo := dao.GetModelProviderManager().FindProvider("OpenAI")
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	code, err := NewModelProviderService().AddModel(
		t.Context(),
		&AddModelRequest{
			ProviderName: "OpenAI",
			InstanceName: "default",
			ModelName:    "remote-model@openai",
			ModelTypes:   []string{"embedding"},
		},
		"user-1",
	)
	if err == nil || !strings.Contains(err.Error(), "catalog unavailable") {
		t.Fatalf("AddModel() error = %v, want catalog error", err)
	}
	if code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}

	var count int64
	if err := db.Model(&entity.TenantModel{}).
		Where("model_name = ?", "remote-model@openai").
		Count(&count).Error; err != nil {
		t.Fatalf("failed to count tenant models: %v", err)
	}
	if count != 0 {
		t.Fatalf("created model count = %d, want 0", count)
	}
}

func TestCreateProviderInstanceRejectsFailedVerificationWithoutMutation(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	probe := &remoteModelProbeDriver{
		DummyModel:    modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		connectionErr: errors.New("credential rejected"),
	}
	installModelServiceProbeDriver(t, "OpenAI", probe)

	code, err := NewModelProviderService().CreateProviderInstance(
		t.Context(),
		"OpenAI",
		"unverified-instance",
		"invalid-key",
		"",
		"",
		"user-1",
		[]CreateInstanceModelInfo{{
			ModelName:  "new-model",
			ModelTypes: []string{"embedding"},
		}},
	)
	if err == nil || !strings.Contains(err.Error(), "credential rejected") {
		t.Fatalf("CreateProviderInstance() error = %v, want verification error", err)
	}
	if code == common.CodeSuccess {
		t.Fatalf("code = %v, want non-success", code)
	}
	if probe.connectionCalls != 1 {
		t.Fatalf("CheckConnection calls = %d, want 1", probe.connectionCalls)
	}

	var instanceCount int64
	if err := db.Model(&entity.TenantModelInstance{}).
		Where("instance_name = ?", "unverified-instance").
		Count(&instanceCount).Error; err != nil {
		t.Fatalf("count provider instances: %v", err)
	}
	if instanceCount != 0 {
		t.Fatalf("created instance count = %d, want 0", instanceCount)
	}
	var modelCount int64
	if err := db.Model(&entity.TenantModel{}).
		Where("model_name = ?", "new-model").
		Count(&modelCount).Error; err != nil {
		t.Fatalf("count provider models: %v", err)
	}
	if modelCount != 0 {
		t.Fatalf("created model count = %d, want 0", modelCount)
	}
}

func TestCreateProviderInstanceRejectsFailedVerificationWithoutModels(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	probe := &remoteModelProbeDriver{
		DummyModel:    modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		connectionErr: errors.New("credential rejected"),
	}
	installModelServiceProbeDriver(t, "OpenAI", probe)

	code, err := NewModelProviderService().CreateProviderInstance(
		t.Context(),
		"OpenAI",
		"unverified-empty-instance",
		"invalid-key",
		"",
		"",
		"user-1",
		nil,
	)
	if err == nil || !strings.Contains(err.Error(), "credential rejected") {
		t.Fatalf("CreateProviderInstance() error = %v, want verification error", err)
	}
	if code == common.CodeSuccess {
		t.Fatalf("code = %v, want non-success", code)
	}
	if probe.connectionCalls != 1 {
		t.Fatalf("CheckConnection calls = %d, want 1", probe.connectionCalls)
	}

	var instanceCount int64
	if err := db.Model(&entity.TenantModelInstance{}).
		Where("instance_name = ?", "unverified-empty-instance").
		Count(&instanceCount).Error; err != nil {
		t.Fatalf("count provider instances: %v", err)
	}
	if instanceCount != 0 {
		t.Fatalf("created instance count = %d, want 0", instanceCount)
	}
}

func TestAlterProviderInstanceRejectsFailedVerificationWithoutMutation(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	probe := &remoteModelProbeDriver{
		DummyModel:    modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		connectionErr: errors.New("credential rejected"),
	}
	installModelServiceProbeDriver(t, "OpenAI", probe)

	code, err := NewModelProviderService().AlterProviderInstance(
		t.Context(),
		"user-1",
		"OpenAI",
		"instance-1",
		"renamed-after-verification",
		"invalid-key",
		"",
		"new-region",
		[]CreateInstanceModelInfo{{
			ModelName:  "replacement-model",
			ModelTypes: []string{"embedding"},
		}},
		true,
	)
	if err == nil || !strings.Contains(err.Error(), "credential rejected") {
		t.Fatalf("AlterProviderInstance() error = %v, want verification error", err)
	}
	if code == common.CodeSuccess {
		t.Fatalf("code = %v, want non-success", code)
	}
	if probe.connectionCalls != 1 {
		t.Fatalf("CheckConnection calls = %d, want 1", probe.connectionCalls)
	}

	var instance entity.TenantModelInstance
	if err := db.Where("id = ?", "instance-1").Take(&instance).Error; err != nil {
		t.Fatalf("reload instance: %v", err)
	}
	if instance.InstanceName != "default" ||
		instance.APIKey != "sk-test" ||
		instance.Extra != "{}" {
		t.Fatalf("instance after failure = %#v, want original values", instance)
	}
	var models []entity.TenantModel
	if err := db.Where("instance_id = ?", "instance-1").Order("id").Find(&models).Error; err != nil {
		t.Fatalf("reload models: %v", err)
	}
	if len(models) != 1 ||
		models[0].ID != "model-1" ||
		models[0].ModelName != "gpt-test" ||
		models[0].ModelType != int(entity.ModelTypeChat) {
		t.Fatalf("models after failure = %#v, want original model", models)
	}
}

func TestAlterProviderInstanceVerifiesWithRetainedEndpointMetadata(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	if err := db.Model(&entity.TenantModelInstance{}).
		Where("id = ?", "instance-1").
		Update("extra", `{"base_url":"https://models.example.test/v1","region":"stored-region"}`).
		Error; err != nil {
		t.Fatalf("seed instance endpoint metadata: %v", err)
	}
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
	}
	installModelServiceProbeDriver(t, "OpenAI", probe)

	code, err := NewModelProviderService().AlterProviderInstance(
		t.Context(),
		"user-1",
		"OpenAI",
		"instance-1",
		"",
		"new-key",
		"",
		"",
		[]CreateInstanceModelInfo{{
			ModelName:  "gpt-test",
			ModelTypes: []string{"chat"},
		}},
		true,
	)
	if err != nil {
		t.Fatalf("AlterProviderInstance() error = %v", err)
	}
	if code != common.CodeSuccess {
		t.Fatalf("code = %v, want %v", code, common.CodeSuccess)
	}
	if probe.connectionCalls != 1 {
		t.Fatalf("CheckConnection calls = %d, want 1", probe.connectionCalls)
	}
	if probe.connectionAPIConfig.BaseURL == nil ||
		*probe.connectionAPIConfig.BaseURL != "https://models.example.test/v1" {
		t.Fatalf(
			"verified base URL = %v, want retained endpoint",
			probe.connectionAPIConfig.BaseURL,
		)
	}
	if probe.connectionAPIConfig.Region == nil ||
		*probe.connectionAPIConfig.Region != "stored-region" {
		t.Fatalf(
			"verified region = %v, want retained region",
			probe.connectionAPIConfig.Region,
		)
	}
}

func TestCreateProviderInstanceRollsBackCatalogFailure(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		listErr:    errors.New("catalog unavailable"),
	}
	probe.newInstanceResult = probe
	providerInfo := dao.GetModelProviderManager().FindProvider("OpenAI")
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	code, err := NewModelProviderService().CreateProviderInstance(
		t.Context(),
		"OpenAI",
		"new-instance",
		"test-key",
		"",
		"",
		"user-1",
		[]CreateInstanceModelInfo{{
			ModelName:  "remote-model@openai",
			ModelTypes: []string{"embedding"},
		}},
	)
	if err == nil || !strings.Contains(err.Error(), "catalog unavailable") {
		t.Fatalf("CreateProviderInstance() error = %v, want catalog error", err)
	}
	if code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}

	var instanceCount int64
	if err := db.Model(&entity.TenantModelInstance{}).
		Where("instance_name = ?", "new-instance").
		Count(&instanceCount).Error; err != nil {
		t.Fatalf("count provider instances: %v", err)
	}
	if instanceCount != 0 {
		t.Fatalf("created instance count = %d, want 0", instanceCount)
	}
}

func TestCreateProviderInstanceRollsBackEarlierModelsOnCatalogFailure(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		listErr:    errors.New("catalog unavailable"),
	}
	probe.newInstanceResult = probe
	providerInfo := dao.GetModelProviderManager().FindProvider("OpenAI")
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	code, err := NewModelProviderService().CreateProviderInstance(
		t.Context(),
		"OpenAI",
		"new-instance",
		"test-key",
		"",
		"",
		"user-1",
		[]CreateInstanceModelInfo{
			{ModelName: "plain-model", ModelTypes: []string{"chat"}},
			{ModelName: "remote-model@openai", ModelTypes: []string{"embedding"}},
		},
	)
	if err == nil || !strings.Contains(err.Error(), "catalog unavailable") {
		t.Fatalf("CreateProviderInstance() error = %v, want catalog error", err)
	}
	if code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}

	var instanceCount int64
	if err := db.Model(&entity.TenantModelInstance{}).
		Where("instance_name = ?", "new-instance").
		Count(&instanceCount).Error; err != nil {
		t.Fatalf("count provider instances: %v", err)
	}
	if instanceCount != 0 {
		t.Fatalf("created instance count = %d, want 0", instanceCount)
	}
	var modelCount int64
	if err := db.Model(&entity.TenantModel{}).
		Where("model_name IN ?", []string{"plain-model", "remote-model@openai"}).
		Count(&modelCount).Error; err != nil {
		t.Fatalf("count provider models: %v", err)
	}
	if modelCount != 0 {
		t.Fatalf("created model count = %d, want 0", modelCount)
	}
}

func TestAlterProviderInstanceRollsBackCatalogFailure(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		listErr:    errors.New("catalog unavailable"),
	}
	probe.newInstanceResult = probe
	providerInfo := dao.GetModelProviderManager().FindProvider("OpenAI")
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	code, err := NewModelProviderService().AlterProviderInstance(
		t.Context(),
		"user-1",
		"OpenAI",
		"instance-1",
		"renamed",
		"new-key",
		"",
		"new-region",
		[]CreateInstanceModelInfo{{
			ModelName:  "remote-model@openai",
			ModelTypes: []string{"embedding"},
		}},
		false,
	)
	if err == nil || !strings.Contains(err.Error(), "catalog unavailable") {
		t.Fatalf("AlterProviderInstance() error = %v, want catalog error", err)
	}
	if code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}

	var instance entity.TenantModelInstance
	if err := db.Where("id = ?", "instance-1").First(&instance).Error; err != nil {
		t.Fatalf("reload instance: %v", err)
	}
	if instance.InstanceName != "default" || instance.APIKey != "sk-test" || instance.Extra != "{}" {
		t.Fatalf("instance after failure = %#v, want original values", instance)
	}
	var model entity.TenantModel
	if err := db.Where("id = ?", "model-1").First(&model).Error; err != nil {
		t.Fatalf("reload retained model: %v", err)
	}
}

func TestAlterProviderInstanceUsesNormalizedSubmittedNames(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		remoteModels: []modelModule.ListModelResponse{{
			Name: "gpt-test",
		}},
	}
	providerInfo := dao.GetModelProviderManager().FindProvider("OpenAI")
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	code, err := NewModelProviderService().AlterProviderInstance(
		t.Context(),
		"user-1",
		"OpenAI",
		"instance-1",
		"",
		"sk-test",
		"",
		"",
		[]CreateInstanceModelInfo{{
			ModelName:  "gpt-test@openai",
			ModelTypes: []string{"chat"},
			MaxTokens:  2048,
		}},
		false,
	)
	if err != nil {
		t.Fatalf("AlterProviderInstance() error = %v", err)
	}
	if code != common.CodeSuccess {
		t.Fatalf("code = %v, want %v", code, common.CodeSuccess)
	}

	var models []entity.TenantModel
	if err := db.Where("instance_id = ?", "instance-1").Find(&models).Error; err != nil {
		t.Fatalf("list instance models: %v", err)
	}
	if len(models) != 1 || models[0].ID != "model-1" || models[0].ModelName != "gpt-test" {
		t.Fatalf("models = %#v, want original normalized model row", models)
	}
}

func TestAlterProviderInstanceRejectsDuplicateInstanceNameAsConflict(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	if err := db.Create(&entity.TenantModelInstance{
		ID:           "instance-2",
		ProviderID:   "provider-1",
		InstanceName: "existing-name",
		Status:       "active",
		Extra:        "{}",
	}).Error; err != nil {
		t.Fatalf("seed conflicting instance: %v", err)
	}

	code, err := NewModelProviderService().AlterProviderInstance(
		t.Context(),
		"user-1",
		"OpenAI",
		"instance-1",
		"existing-name",
		"sk-test",
		"",
		"",
		[]CreateInstanceModelInfo{{
			ModelName:  "gpt-test",
			ModelTypes: []string{"chat"},
		}},
		false,
	)
	if err == nil || !errors.Is(err, dao.ErrTenantModelInstanceExists) {
		t.Fatalf(
			"AlterProviderInstance() error = %v, want duplicate instance error",
			err,
		)
	}
	if code != common.CodeConflict {
		t.Fatalf("code = %v, want %v", code, common.CodeConflict)
	}

	var original entity.TenantModelInstance
	if err := db.Where("id = ?", "instance-1").First(&original).Error; err != nil {
		t.Fatalf("reload original instance: %v", err)
	}
	if original.InstanceName != "default" {
		t.Fatalf(
			"original instance name = %q, want unchanged default",
			original.InstanceName,
		)
	}
}

func TestCreateProviderInstanceRejectsNormalizedDuplicateNames(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		remoteModels: []modelModule.ListModelResponse{{
			Name: "same-model",
		}},
	}
	providerInfo := dao.GetModelProviderManager().FindProvider("OpenAI")
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	code, err := NewModelProviderService().CreateProviderInstance(
		t.Context(),
		"OpenAI",
		"new-instance",
		"test-key",
		"",
		"",
		"user-1",
		[]CreateInstanceModelInfo{
			{ModelName: "same-model", ModelTypes: []string{"chat"}},
			{ModelName: "same-model@openai", ModelTypes: []string{"embedding"}},
		},
	)
	if err == nil || !strings.Contains(err.Error(), "duplicate normalized model") {
		t.Fatalf("CreateProviderInstance() error = %v, want normalized duplicate error", err)
	}
	if code != common.CodeConflict {
		t.Fatalf("code = %v, want %v", code, common.CodeConflict)
	}
	var count int64
	if err := db.Model(&entity.TenantModelInstance{}).
		Where("instance_name = ?", "new-instance").
		Count(&count).Error; err != nil {
		t.Fatalf("count provider instances: %v", err)
	}
	if count != 0 {
		t.Fatalf("created instance count = %d, want 0", count)
	}
}

func TestCreateProviderInstanceClassifiesUnlistedRemoteModelAsDataError(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		remoteModels: []modelModule.ListModelResponse{{
			Name: "different-model",
		}},
	}
	probe.newInstanceResult = probe
	providerInfo := dao.GetModelProviderManager().FindProvider("OpenAI")
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	code, err := NewModelProviderService().CreateProviderInstance(
		t.Context(),
		"OpenAI",
		"new-instance",
		"test-key",
		"",
		"",
		"user-1",
		[]CreateInstanceModelInfo{{
			ModelName:  "remote-model@openai",
			ModelTypes: []string{"embedding"},
		}},
	)
	if err == nil || !strings.Contains(err.Error(), "not found in remote model catalog") {
		t.Fatalf("CreateProviderInstance() error = %v, want unlisted model error", err)
	}
	if code != common.CodeDataError {
		t.Fatalf("code = %v, want %v", code, common.CodeDataError)
	}
}

func TestCreateProviderInstanceRejectsExistingInstanceAsConflict(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)

	code, err := NewModelProviderService().CreateProviderInstance(
		t.Context(),
		"OpenAI",
		"default",
		"test-key",
		"",
		"",
		"user-1",
		nil,
	)
	if err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("CreateProviderInstance() error = %v, want duplicate instance error", err)
	}
	if code != common.CodeConflict {
		t.Fatalf("code = %v, want %v", code, common.CodeConflict)
	}
}

func TestCreateNameOnlyProviderInstanceRejectsExistingInstanceAsConflict(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	if err := db.Create(&entity.TenantModelInstance{
		ID:           "instance-existing",
		ProviderID:   "provider-1",
		InstanceName: "existing-name",
		Status:       "active",
		Extra:        "{}",
	}).Error; err != nil {
		t.Fatalf("seed existing instance: %v", err)
	}

	code, err := NewModelProviderService().CreateNameOnlyProviderInstance(
		t.Context(),
		"OpenAI",
		"existing-name",
		"user-1",
	)
	if err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("CreateNameOnlyProviderInstance() error = %v, want duplicate instance error", err)
	}
	if code != common.CodeConflict {
		t.Fatalf("code = %v, want %v", code, common.CodeConflict)
	}
}

func TestCreateProviderInstanceRejectsBlankModelNameAsBadRequest(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)

	code, err := NewModelProviderService().CreateProviderInstance(
		t.Context(),
		"OpenAI",
		"new-instance",
		"test-key",
		"",
		"",
		"user-1",
		[]CreateInstanceModelInfo{{
			ModelName:  "",
			ModelTypes: []string{"chat"},
		}},
	)
	if err == nil || !strings.Contains(err.Error(), "model name is required") {
		t.Fatalf("CreateProviderInstance() error = %v, want required-name error", err)
	}
	if code != common.CodeBadRequest {
		t.Fatalf("code = %v, want %v", code, common.CodeBadRequest)
	}
}

func TestAlterProviderInstanceRejectsBlankModelNameAsBadRequest(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)

	code, err := NewModelProviderService().AlterProviderInstance(
		t.Context(),
		"user-1",
		"OpenAI",
		"instance-1",
		"",
		"sk-test",
		"",
		"",
		[]CreateInstanceModelInfo{{
			ModelName:  "",
			ModelTypes: []string{"chat"},
		}},
		false,
	)
	if err == nil || !strings.Contains(err.Error(), "model name is required") {
		t.Fatalf("AlterProviderInstance() error = %v, want required-name error", err)
	}
	if code != common.CodeBadRequest {
		t.Fatalf("code = %v, want %v", code, common.CodeBadRequest)
	}
}

func TestAlterProviderInstanceRejectsNormalizedDuplicateNamesAsConflict(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	probe := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		remoteModels: []modelModule.ListModelResponse{{
			Name: "same-model",
		}},
	}
	providerInfo := dao.GetModelProviderManager().FindProvider("OpenAI")
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = probe
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	code, err := NewModelProviderService().AlterProviderInstance(
		t.Context(),
		"user-1",
		"OpenAI",
		"instance-1",
		"",
		"sk-test",
		"",
		"",
		[]CreateInstanceModelInfo{
			{ModelName: "same-model", ModelTypes: []string{"chat"}},
			{ModelName: "same-model@openai", ModelTypes: []string{"embedding"}},
		},
		false,
	)
	if err == nil || !strings.Contains(err.Error(), "duplicate normalized model") {
		t.Fatalf("AlterProviderInstance() error = %v, want normalized duplicate error", err)
	}
	if code != common.CodeConflict {
		t.Fatalf("code = %v, want %v", code, common.CodeConflict)
	}
}

func TestAlterProviderInstanceRejectsMissingInstanceUpdateRow(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	if err := db.Exec(`
		CREATE TRIGGER remove_model_instance_before_update
		BEFORE UPDATE ON tenant_model_instance
		BEGIN
			DELETE FROM tenant_model_instance WHERE id = OLD.id;
			SELECT RAISE(IGNORE);
		END
	`).Error; err != nil {
		t.Fatalf("create instance update trigger: %v", err)
	}

	code, err := NewModelProviderService().AlterProviderInstance(
		t.Context(), "user-1", "OpenAI", "instance-1", "", "new-key", "", "",
		[]CreateInstanceModelInfo{{ModelName: "gpt-test", ModelTypes: []string{"chat"}}}, false,
	)
	if err == nil || !strings.Contains(err.Error(), "updated 0 rows") {
		t.Fatalf("AlterProviderInstance() error = %v, want zero-row instance update error", err)
	}
	if code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}
	var count int64
	if err := db.Model(&entity.TenantModelInstance{}).Where("id = ?", "instance-1").Count(&count).Error; err != nil {
		t.Fatalf("count instance after rollback: %v", err)
	}
	if count != 1 {
		t.Fatalf("instance count after rollback = %d, want 1", count)
	}
}

func TestAlterProviderInstanceRejectsMissingModelUpdateRow(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	if err := db.Exec(`
		CREATE TRIGGER remove_model_before_provider_update
		BEFORE UPDATE ON tenant_model
		BEGIN
			DELETE FROM tenant_model WHERE id = OLD.id;
			SELECT RAISE(IGNORE);
		END
	`).Error; err != nil {
		t.Fatalf("create model update trigger: %v", err)
	}

	code, err := NewModelProviderService().AlterProviderInstance(
		t.Context(), "user-1", "OpenAI", "instance-1", "", "new-key", "", "",
		[]CreateInstanceModelInfo{{ModelName: "gpt-test", ModelTypes: []string{"embedding"}}}, false,
	)
	if err == nil || !strings.Contains(err.Error(), "updated 0 rows") {
		t.Fatalf("AlterProviderInstance() error = %v, want zero-row model update error", err)
	}
	if code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}
	var instance entity.TenantModelInstance
	if err := db.Where("id = ?", "instance-1").First(&instance).Error; err != nil {
		t.Fatalf("reload instance after rollback: %v", err)
	}
	if instance.APIKey != "sk-test" {
		t.Fatalf("instance API key changed despite rollback")
	}
	var model entity.TenantModel
	if err := db.Where("id = ?", "model-1").First(&model).Error; err != nil {
		t.Fatalf("reload model after rollback: %v", err)
	}
	if model.ModelType != int(entity.ModelTypeChat) {
		t.Fatalf("model type = %d, want unchanged chat", model.ModelType)
	}
}

func TestAlterProviderInstanceRejectsChangedDeleteCount(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	if err := db.Exec(`
		CREATE TRIGGER ignore_model_delete
		BEFORE DELETE ON tenant_model
		BEGIN
			SELECT RAISE(IGNORE);
		END
	`).Error; err != nil {
		t.Fatalf("create model delete trigger: %v", err)
	}

	code, err := NewModelProviderService().AlterProviderInstance(
		t.Context(), "user-1", "OpenAI", "instance-1", "", "new-key", "", "", nil, false,
	)
	if err == nil || !strings.Contains(err.Error(), "deleted 0 rows") {
		t.Fatalf("AlterProviderInstance() error = %v, want delete-count error", err)
	}
	if code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}
	var instance entity.TenantModelInstance
	if err := db.Where("id = ?", "instance-1").First(&instance).Error; err != nil {
		t.Fatalf("reload instance after rollback: %v", err)
	}
	if instance.APIKey != "sk-test" {
		t.Fatalf("instance API key changed despite rollback")
	}
	var modelCount int64
	if err := db.Model(&entity.TenantModel{}).Where("id = ?", "model-1").Count(&modelCount).Error; err != nil {
		t.Fatalf("count model after rollback: %v", err)
	}
	if modelCount != 1 {
		t.Fatalf("model count after rollback = %d, want 1", modelCount)
	}
}

func TestValidateEmbeddingModel(t *testing.T) {
	maxDimension := 2048
	maxBatchSize := 128

	tests := []struct {
		name               string
		model              *modelModule.Model
		requestedDimension int
		requestedBatchSize int
		wantErr            string
	}{
		{
			name:               "rejects nil model",
			requestedDimension: 1024,
			requestedBatchSize: 16,
			wantErr:            "embedding model is nil",
		},
		{
			name:               "rejects zero dimension",
			model:              &modelModule.Model{},
			requestedDimension: 0,
			requestedBatchSize: 1,
			wantErr:            "input dimension <= 0",
		},
		{
			name:               "rejects negative dimension",
			model:              &modelModule.Model{},
			requestedDimension: -1,
			requestedBatchSize: 1,
			wantErr:            "input dimension <= 0",
		},
		{
			name:               "rejects zero batch size",
			model:              &modelModule.Model{},
			requestedDimension: 1024,
			requestedBatchSize: 0,
			wantErr:            "input batch size <= 0",
		},
		{
			name:               "rejects negative batch size",
			model:              &modelModule.Model{},
			requestedDimension: 1024,
			requestedBatchSize: -1,
			wantErr:            "input batch size <= 0",
		},
		{
			name:               "allows unknown max dimension",
			model:              &modelModule.Model{MaxBatchSize: &maxBatchSize},
			requestedDimension: 1024,
			requestedBatchSize: 1,
		},
		{
			name:               "allows unknown max batch size",
			model:              &modelModule.Model{MaxDimension: &maxDimension},
			requestedDimension: 1024,
			requestedBatchSize: 1,
		},
		{
			name:               "allows unknown embedding limits",
			model:              &modelModule.Model{Name: "custom-embedding"},
			requestedDimension: 1024,
			requestedBatchSize: 1,
		},
		{
			name:               "allows dimension listed in explicit options",
			model:              &modelModule.Model{Name: "embedding-3", MaxDimension: &maxDimension, MaxBatchSize: &maxBatchSize, Dimensions: []int{256, 512, 1024, 2048}},
			requestedDimension: 1024,
			requestedBatchSize: 128,
		},
		{
			name:               "rejects dimension not listed in explicit options",
			model:              &modelModule.Model{Name: "embedding-3", MaxDimension: &maxDimension, MaxBatchSize: &maxBatchSize, Dimensions: []int{256, 512, 1024, 2048}},
			requestedDimension: 1536,
			requestedBatchSize: 128,
			wantErr:            "supported dimensions",
		},
		{
			name:               "allows custom dimension within max dimension",
			model:              &modelModule.Model{Name: "flex-embedding", MaxDimension: &maxDimension, MaxBatchSize: &maxBatchSize},
			requestedDimension: 1536,
			requestedBatchSize: 1,
		},
		{
			name:               "rejects custom dimension above max dimension",
			model:              &modelModule.Model{Name: "flex-embedding", MaxDimension: &maxDimension, MaxBatchSize: &maxBatchSize},
			requestedDimension: 4096,
			requestedBatchSize: 1,
			wantErr:            "max dimension",
		},
		{
			name:               "allows batch at model limit",
			model:              &modelModule.Model{Name: "embedding-3", MaxDimension: &maxDimension, MaxBatchSize: &maxBatchSize},
			requestedDimension: 1024,
			requestedBatchSize: 128,
		},
		{
			name:               "rejects batch above model limit",
			model:              &modelModule.Model{Name: "embedding-3", MaxDimension: &maxDimension, MaxBatchSize: &maxBatchSize},
			requestedDimension: 1024,
			requestedBatchSize: 129,
			wantErr:            "max batch size",
		},
		{
			name:               "rejects batch when model limit is unspecified",
			model:              &modelModule.Model{Name: "custom-embedding", MaxDimension: &maxDimension},
			requestedDimension: 1024,
			requestedBatchSize: 10000,
			wantErr:            "max batch size is nil",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateEmbeddingModel(tt.model, tt.requestedDimension, tt.requestedBatchSize)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("validateEmbeddingModel() error = %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("validateEmbeddingModel() expected error containing %q", tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("validateEmbeddingModel() error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestVerifyProviderModelValidatesRemoteEmbeddingMetadata(t *testing.T) {
	maxDimension := 1024
	maxBatchSize := 0
	driver := &remoteModelProbeDriver{
		DummyModel: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
		remoteModels: []modelModule.ListModelResponse{{
			Name:         "remote-embedding",
			ModelTypes:   []string{"embedding"},
			MaxDimension: &maxDimension,
			MaxBatchSize: &maxBatchSize,
		}},
	}

	result, err := verifyProviderModel(context.Background(), driver, nil, &modelModule.APIConfig{}, nil)
	if err == nil {
		t.Fatal("verifyProviderModel() error = nil, want validation error")
	}
	if result["remote-embedding"] != entity.ModelVerifyFail {
		t.Fatalf("verification result = %#v, want remote model failure", result)
	}
	if driver.embedCalls != 0 {
		t.Fatalf("Embed calls = %d, want 0 after metadata validation failure", driver.embedCalls)
	}
}

func TestModelInfoWithTenantExtraAppliesEmbeddingConstraints(t *testing.T) {
	factoryMaxDimension := 2048
	factoryBatchSize := 128
	modelInfo := &modelModule.Model{
		Name:         "embedding-3",
		MaxDimension: &factoryMaxDimension,
		MaxBatchSize: &factoryBatchSize,
		Dimensions:   []int{1024, 2048},
		ModelTypes:   []string{"embedding"},
		ModelTypeMap: map[string]bool{"embedding": true},
	}
	modelEntity := &entity.TenantModel{
		Extra: `{"max_dimension":768,"max_batch_size":16,"dimensions":[384,768],"model_types":["embedding"]}`,
	}

	merged, err := modelInfoWithTenantExtra(modelInfo, modelEntity)
	if err != nil {
		t.Fatalf("modelInfoWithTenantExtra() error = %v", err)
	}
	if merged == modelInfo {
		t.Fatalf("modelInfoWithTenantExtra() returned original model pointer")
	}
	if merged.MaxDimension == nil || *merged.MaxDimension != 768 {
		t.Fatalf("MaxDimension = %v, want 768", merged.MaxDimension)
	}
	if merged.MaxBatchSize == nil || *merged.MaxBatchSize != 16 {
		t.Fatalf("MaxBatchSize = %v, want 16", merged.MaxBatchSize)
	}
	if len(merged.Dimensions) != 2 || merged.Dimensions[0] != 384 || merged.Dimensions[1] != 768 {
		t.Fatalf("Dimensions = %v, want [384 768]", merged.Dimensions)
	}
	if validationErr := validateEmbeddingModel(merged, 1024, 16); validationErr == nil || !strings.Contains(validationErr.Error(), "supported dimensions") {
		t.Fatalf("validateEmbeddingModel() error = %v, want supported dimensions error", validationErr)
	}
	if validationErr := validateEmbeddingModel(merged, 768, 16); validationErr != nil {
		t.Fatalf("validateEmbeddingModel() error = %v", validationErr)
	}
	if validationErr := validateEmbeddingModel(merged, 768, 17); validationErr == nil || !strings.Contains(validationErr.Error(), "max batch size") {
		t.Fatalf("validateEmbeddingModel() error = %v, want max batch size error", validationErr)
	}
	if modelInfo.MaxDimension == nil || *modelInfo.MaxDimension != factoryMaxDimension {
		t.Fatalf("factory MaxDimension was mutated: %v", modelInfo.MaxDimension)
	}
	if modelInfo.MaxBatchSize == nil || *modelInfo.MaxBatchSize != factoryBatchSize {
		t.Fatalf("factory MaxBatchSize was mutated: %v", modelInfo.MaxBatchSize)
	}
	if len(modelInfo.Dimensions) != 2 || modelInfo.Dimensions[0] != 1024 || modelInfo.Dimensions[1] != 2048 {
		t.Fatalf("factory Dimensions were mutated: %v", modelInfo.Dimensions)
	}
}

func setupModelProviderServiceTestDB(t *testing.T) *gorm.DB {
	t.Helper()
	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{TranslateError: true})
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	if err := db.AutoMigrate(
		&entity.UserTenant{},
		&entity.TenantModelProvider{},
		&entity.TenantModelInstance{},
		&entity.TenantModel{},
	); err != nil {
		t.Fatalf("failed to migrate model service tables: %v", err)
	}
	return db
}

func useModelProviderServiceTestDB(t *testing.T, db *gorm.DB) {
	t.Helper()
	orig := dao.DB
	dao.DB = db
	t.Cleanup(func() { dao.DB = orig })
}

func seedModelProviderServiceScope(t *testing.T, db *gorm.DB) {
	t.Helper()
	activeStatus := "1"
	rows := []interface{}{
		&entity.UserTenant{ID: "user-tenant-1", UserID: "user-1", TenantID: "tenant-1", Role: "owner", InvitedBy: "user-1", Status: &activeStatus},
		&entity.TenantModelProvider{ID: "provider-1", TenantID: "tenant-1", ProviderName: "OpenAI"},
		&entity.TenantModelInstance{ID: "instance-1", ProviderID: "provider-1", InstanceName: "default", APIKey: "sk-test", Status: "active", Extra: "{}"},
		&entity.TenantModel{ID: "model-1", ProviderID: "provider-1", InstanceID: "instance-1", ModelName: "gpt-test", ModelType: int(entity.ModelTypeChat), Status: "active"},
	}
	for _, row := range rows {
		if err := db.Create(row).Error; err != nil {
			t.Fatalf("failed to seed %T: %v", row, err)
		}
	}
}

func TestModelProviderServiceAlterModelStatusByID(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)

	ctx := t.Context()
	code, err := NewModelProviderService().AlterModel(ctx, "OpenAI", "default", "", "user-1", "model-1", map[string]interface{}{"status": "inactive"})
	if err != nil {
		t.Fatalf("AlterModel() error = %v", err)
	}
	if code != common.CodeSuccess {
		t.Fatalf("code = %v, want %v", code, common.CodeSuccess)
	}

	var got entity.TenantModel
	if err := db.Where("id = ?", "model-1").First(&got).Error; err != nil {
		t.Fatalf("failed to reload tenant model: %v", err)
	}
	if got.Status != "inactive" {
		t.Fatalf("status = %q, want inactive", got.Status)
	}
}

func TestModelProviderServiceAlterModelDoesNotPartiallyPersistFailedUpdate(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	if err := db.Exec(`
		CREATE TRIGGER fail_tenant_model_metadata_update
		BEFORE UPDATE OF model_type, extra ON tenant_model
		BEGIN
			SELECT RAISE(FAIL, 'forced metadata update failure');
		END
	`).Error; err != nil {
		t.Fatalf("failed to create update trigger: %v", err)
	}

	code, err := NewModelProviderService().AlterModel(
		t.Context(),
		"OpenAI",
		"default",
		"",
		"user-1",
		"model-1",
		map[string]interface{}{
			"status":     "inactive",
			"model_type": "embedding",
			"max_tokens": 2048,
		},
	)
	if err == nil || !strings.Contains(err.Error(), "forced metadata update failure") {
		t.Fatalf("AlterModel() error = %v, want forced metadata update failure", err)
	}
	if code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}

	var got entity.TenantModel
	if err := db.Where("id = ?", "model-1").First(&got).Error; err != nil {
		t.Fatalf("failed to reload tenant model: %v", err)
	}
	if got.Status != "active" {
		t.Fatalf("status = %q, want unchanged active status", got.Status)
	}
	if got.ModelType != int(entity.ModelTypeChat) {
		t.Fatalf("model_type = %d, want unchanged %d", got.ModelType, entity.ModelTypeChat)
	}
}

func TestModelProviderServiceAlterModelRejectsZeroUpdatedRows(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	if err := db.Exec(`
		CREATE TRIGGER remove_tenant_model_before_update
		BEFORE UPDATE ON tenant_model
		BEGIN
			DELETE FROM tenant_model WHERE id = OLD.id;
			SELECT RAISE(IGNORE);
		END
	`).Error; err != nil {
		t.Fatalf("failed to create update trigger: %v", err)
	}

	code, err := NewModelProviderService().AlterModel(
		t.Context(),
		"OpenAI",
		"default",
		"",
		"user-1",
		"model-1",
		map[string]interface{}{"status": "inactive"},
	)
	if err == nil || !strings.Contains(err.Error(), "updated 0 rows") {
		t.Fatalf("AlterModel() error = %v, want zero-row update error", err)
	}
	if code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}
}

func TestModelProviderServiceAlterModelStatusByNameRejectsAmbiguousRows(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	duplicate := &entity.TenantModel{
		ID:         "model-duplicate",
		ProviderID: "provider-1",
		InstanceID: "instance-1",
		ModelName:  "gpt-test",
		ModelType:  int(entity.ModelTypeEmbedding),
		Status:     "active",
	}
	if err := db.Create(duplicate).Error; err != nil {
		t.Fatalf("failed to seed duplicate tenant model: %v", err)
	}

	code, err := NewModelProviderService().AlterModel(
		t.Context(),
		"OpenAI",
		"default",
		"gpt-test",
		"user-1",
		"",
		map[string]interface{}{"status": "inactive"},
	)
	if err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("AlterModel() error = %v, want ambiguity error", err)
	}
	if code != common.CodeConflict {
		t.Fatalf("code = %v, want %v", code, common.CodeConflict)
	}

	var rows []entity.TenantModel
	if err := db.Where("provider_id = ? AND instance_id = ? AND model_name = ?", "provider-1", "instance-1", "gpt-test").Find(&rows).Error; err != nil {
		t.Fatalf("failed to reload duplicate models: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("row count = %d, want 2", len(rows))
	}
	for _, row := range rows {
		if row.Status != "active" {
			t.Fatalf("model %q status = %q, want unchanged active status", row.ID, row.Status)
		}
	}
}

func TestModelProviderServiceAlterModelStatusByNameAndIDDisambiguatesRows(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	duplicate := &entity.TenantModel{
		ID:         "model-duplicate",
		ProviderID: "provider-1",
		InstanceID: "instance-1",
		ModelName:  "gpt-test",
		ModelType:  int(entity.ModelTypeEmbedding),
		Status:     "active",
	}
	if err := db.Create(duplicate).Error; err != nil {
		t.Fatalf("failed to seed duplicate tenant model: %v", err)
	}

	code, err := NewModelProviderService().AlterModel(
		t.Context(),
		"OpenAI",
		"default",
		"gpt-test",
		"user-1",
		duplicate.ID,
		map[string]interface{}{"status": "inactive"},
	)
	if err != nil {
		t.Fatalf("AlterModel() error = %v", err)
	}
	if code != common.CodeSuccess {
		t.Fatalf("code = %v, want %v", code, common.CodeSuccess)
	}

	var rows []entity.TenantModel
	if err := db.Where("provider_id = ? AND instance_id = ? AND model_name = ?", "provider-1", "instance-1", "gpt-test").Order("id").Find(&rows).Error; err != nil {
		t.Fatalf("failed to reload duplicate models: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("row count = %d, want 2", len(rows))
	}
	statuses := make(map[string]string, len(rows))
	for _, row := range rows {
		statuses[row.ID] = row.Status
	}
	if statuses["model-1"] != "active" || statuses[duplicate.ID] != "inactive" {
		t.Fatalf("statuses = %#v, want only %q inactive", statuses, duplicate.ID)
	}
}

func TestModelProviderServiceGetModelConfigByID(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)

	ctx := t.Context()
	driver, modelName, apiConfig, _, err := NewModelProviderService().GetModelConfigByID(ctx, "user-1", entity.ModelTypeChat, "model-1")
	if err != nil {
		t.Fatalf("GetModelConfigByID() error = %v", err)
	}
	if driver == nil {
		t.Fatal("GetModelConfigByID() returned nil driver")
	}
	if modelName != "gpt-test" {
		t.Fatalf("modelName = %q, want %q", modelName, "gpt-test")
	}
	if apiConfig == nil || apiConfig.ApiKey == nil || *apiConfig.ApiKey != "sk-test" {
		t.Fatalf("apiConfig.ApiKey = %v, want %q", apiConfig.ApiKey, "sk-test")
	}
}

func TestGetModelInstanceAndProviderByIDAllowsCustomTenantModel(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)

	custom := &entity.TenantModel{
		ID:         "custom-embedding",
		ProviderID: "provider-1",
		InstanceID: "instance-1",
		ModelName:  "tenant-only-embedding",
		ModelType:  int(entity.ModelTypeEmbedding),
		Status:     "active",
	}
	if err := db.Create(custom).Error; err != nil {
		t.Fatalf("failed to seed custom model: %v", err)
	}

	modelID := custom.ID
	info, err := NewModelProviderService().getModelInstanceAndProviderByID(
		t.Context(),
		&modelID,
		"user-1",
		&modelModule.APIConfig{},
	)
	if err != nil {
		t.Fatalf("getModelInstanceAndProviderByID() error = %v", err)
	}
	if info == nil || info.ModelInfo == nil {
		t.Fatal("getModelInstanceAndProviderByID() returned no model metadata")
	}
	if info.ModelInfo.Name != custom.ModelName {
		t.Fatalf("model name = %q, want %q", info.ModelInfo.Name, custom.ModelName)
	}
	if !info.ModelInfo.ModelTypeMap["embedding"] {
		t.Fatalf("model types = %#v, want embedding", info.ModelInfo.ModelTypeMap)
	}
	if info.ModelInfo.MaxDimension != nil || info.ModelInfo.MaxBatchSize != nil {
		t.Fatalf("custom model limits = (%v, %v), want unknown", info.ModelInfo.MaxDimension, info.ModelInfo.MaxBatchSize)
	}
}

func TestGetModelInstanceAndProviderByNameReturnsUnexpectedLookupErrors(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	if err := db.Migrator().DropTable(&entity.TenantModel{}); err != nil {
		t.Fatalf("failed to break tenant model lookup: %v", err)
	}

	providerName := "OpenAI"
	instanceName := "default"
	modelName := "text-embedding-3-small"
	_, err := NewModelProviderService().getModelInstanceAndProviderByName(
		t.Context(),
		&providerName,
		&instanceName,
		&modelName,
		"user-1",
		&modelModule.APIConfig{},
	)
	if err == nil || !strings.Contains(err.Error(), "lookup failed") {
		t.Fatalf("getModelInstanceAndProviderByName() error = %v, want lookup failure", err)
	}
}

func TestEmbedTextMalformedInstanceMetadataReturnsServerError(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	if err := db.Model(&entity.TenantModelInstance{}).
		Where("id = ?", "instance-1").
		Update("extra", "{").Error; err != nil {
		t.Fatalf("failed to corrupt instance metadata: %v", err)
	}

	modelID := "model-1"
	response, code, err := NewModelProviderService().EmbedText(
		t.Context(),
		nil,
		nil,
		nil,
		&modelID,
		"user-1",
		[]string{"test"},
		&modelModule.APIConfig{},
		&modelModule.EmbeddingConfig{Dimension: 1536},
	)
	if err == nil {
		t.Fatal("EmbedText() error = nil, want malformed metadata error")
	}
	if code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}
	if response != nil {
		t.Fatalf("response = %#v, want nil", response)
	}
}

func TestEmbedTextWithoutOwnerTenantReturnsNotFoundError(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)

	modelID := "model-1"
	response, code, err := NewModelProviderService().EmbedText(
		t.Context(),
		nil,
		nil,
		nil,
		&modelID,
		"user-without-tenant",
		[]string{"test"},
		&modelModule.APIConfig{},
		&modelModule.EmbeddingConfig{Dimension: 1536},
	)
	if err == nil || !strings.Contains(err.Error(), "no tenant") {
		t.Fatalf("EmbedText() error = %v, want no-tenant error", err)
	}
	if code != common.CodeNotFound {
		t.Fatalf("code = %v, want %v", code, common.CodeNotFound)
	}
	if response != nil {
		t.Fatalf("response = %#v, want nil", response)
	}
}

func TestModelResolutionFailureRejectsNilResult(t *testing.T) {
	err := modelResolutionFailure(nil, nil)
	if err == nil || !strings.Contains(err.Error(), "empty model resolution result") {
		t.Fatalf("modelResolutionFailure() error = %v, want invariant error", err)
	}
	if code := modelResolutionErrorCode(err); code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}
}

func TestEmbedTextClassifiesMissingNamedModelAsNotFound(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)

	providerName := "OpenAI"
	instanceName := "default"
	modelName := "tenant-model-that-does-not-exist"
	response, code, err := NewModelProviderService().EmbedText(
		t.Context(),
		&providerName,
		&instanceName,
		&modelName,
		nil,
		"user-1",
		[]string{"test"},
		&modelModule.APIConfig{},
		&modelModule.EmbeddingConfig{Dimension: 1536},
	)
	if err == nil || !errors.Is(err, errTenantModelNotFound) {
		t.Fatalf("EmbedText() error = %v, want not-found error", err)
	}
	if code != common.CodeNotFound {
		t.Fatalf("code = %v, want %v", code, common.CodeNotFound)
	}
	if response != nil {
		t.Fatalf("response = %#v, want nil", response)
	}
}

func TestEmbedTextClassifiesUnexpectedModelLookupErrorAsServerError(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	if err := db.Migrator().DropTable(&entity.TenantModel{}); err != nil {
		t.Fatalf("failed to break tenant model lookup: %v", err)
	}

	providerName := "OpenAI"
	instanceName := "default"
	modelName := "text-embedding-3-small"
	_, code, err := NewModelProviderService().EmbedText(
		t.Context(),
		&providerName,
		&instanceName,
		&modelName,
		nil,
		"user-1",
		[]string{"test"},
		&modelModule.APIConfig{},
		&modelModule.EmbeddingConfig{Dimension: 1536},
	)
	if err == nil || !strings.Contains(err.Error(), "lookup failed") {
		t.Fatalf("EmbedText() error = %v, want lookup failure", err)
	}
	if code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}
}

func TestEmbedTextClassifiesUnexpectedModelIDLookupErrorAsServerError(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	if err := db.Migrator().DropTable(&entity.TenantModel{}); err != nil {
		t.Fatalf("failed to break tenant model lookup: %v", err)
	}

	modelID := "model-1"
	_, code, err := NewModelProviderService().EmbedText(
		t.Context(),
		nil,
		nil,
		nil,
		&modelID,
		"user-1",
		[]string{"test"},
		&modelModule.APIConfig{},
		&modelModule.EmbeddingConfig{Dimension: 1536},
	)
	if err == nil || !strings.Contains(err.Error(), "lookup failed") {
		t.Fatalf("EmbedText() error = %v, want lookup failure", err)
	}
	if code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}
}

func TestEmbedTextClassifiesNameScopeLookupErrorsAsServerError(t *testing.T) {
	tests := []struct {
		name  string
		table interface{}
	}{
		{name: "user tenant", table: &entity.UserTenant{}},
		{name: "provider", table: &entity.TenantModelProvider{}},
		{name: "instance", table: &entity.TenantModelInstance{}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			db := setupModelProviderServiceTestDB(t)
			useModelProviderServiceTestDB(t, db)
			seedModelProviderServiceScope(t, db)
			if err := db.Migrator().DropTable(test.table); err != nil {
				t.Fatalf("failed to break %s lookup: %v", test.name, err)
			}

			providerName := "OpenAI"
			instanceName := "default"
			modelName := "text-embedding-3-small"
			_, code, err := NewModelProviderService().EmbedText(
				t.Context(),
				&providerName,
				&instanceName,
				&modelName,
				nil,
				"user-1",
				[]string{"test"},
				&modelModule.APIConfig{},
				&modelModule.EmbeddingConfig{Dimension: 1536},
			)
			if err == nil {
				t.Fatalf("EmbedText() error = nil, want %s lookup failure", test.name)
			}
			if code != common.CodeServerError {
				t.Fatalf("code = %v, want %v", code, common.CodeServerError)
			}
		})
	}
}

func TestEmbedTextClassifiesModelIDScopeLookupErrorsAsServerError(t *testing.T) {
	tests := []struct {
		name  string
		table interface{}
	}{
		{name: "user tenant", table: &entity.UserTenant{}},
		{name: "instance", table: &entity.TenantModelInstance{}},
		{name: "provider", table: &entity.TenantModelProvider{}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			db := setupModelProviderServiceTestDB(t)
			useModelProviderServiceTestDB(t, db)
			seedModelProviderServiceScope(t, db)
			if err := db.Migrator().DropTable(test.table); err != nil {
				t.Fatalf("failed to break %s lookup: %v", test.name, err)
			}

			modelID := "model-1"
			_, code, err := NewModelProviderService().EmbedText(
				t.Context(),
				nil,
				nil,
				nil,
				&modelID,
				"user-1",
				[]string{"test"},
				&modelModule.APIConfig{},
				&modelModule.EmbeddingConfig{Dimension: 1536},
			)
			if err == nil {
				t.Fatalf("EmbedText() error = nil, want %s lookup failure", test.name)
			}
			if code != common.CodeServerError {
				t.Fatalf("code = %v, want %v", code, common.CodeServerError)
			}
		})
	}
}

func TestEmbedTextReportsAmbiguousTenantModelsAsConflict(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	modelName := "shared-custom-model"
	for _, model := range []*entity.TenantModel{
		{
			ID:         "shared-chat",
			ProviderID: "provider-1",
			InstanceID: "instance-1",
			ModelName:  modelName,
			ModelType:  int(entity.ModelTypeChat),
			Status:     "active",
		},
		{
			ID:         "shared-embedding",
			ProviderID: "provider-1",
			InstanceID: "instance-1",
			ModelName:  modelName,
			ModelType:  int(entity.ModelTypeEmbedding),
			Status:     "active",
		},
	} {
		if err := db.Create(model).Error; err != nil {
			t.Fatalf("failed to seed tenant model: %v", err)
		}
	}

	providerName := "OpenAI"
	instanceName := "default"
	response, code, err := NewModelProviderService().EmbedText(
		t.Context(),
		&providerName,
		&instanceName,
		&modelName,
		nil,
		"user-1",
		[]string{"test"},
		&modelModule.APIConfig{},
		&modelModule.EmbeddingConfig{Dimension: 1536},
	)
	if err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("EmbedText() error = %v, want ambiguity error", err)
	}
	if code != common.CodeConflict {
		t.Fatalf("code = %v, want %v", code, common.CodeConflict)
	}
	if response != nil {
		t.Fatalf("response = %#v, want nil", response)
	}
}

func TestGetModelInstanceAndProviderByNameRejectsAmbiguousTenantModels(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	modelName := "shared-custom-model"
	for _, model := range []*entity.TenantModel{
		{
			ID:         "shared-chat",
			ProviderID: "provider-1",
			InstanceID: "instance-1",
			ModelName:  modelName,
			ModelType:  int(entity.ModelTypeChat),
			Status:     "active",
		},
		{
			ID:         "shared-embedding",
			ProviderID: "provider-1",
			InstanceID: "instance-1",
			ModelName:  modelName,
			ModelType:  int(entity.ModelTypeEmbedding),
			Status:     "active",
		},
	} {
		if err := db.Create(model).Error; err != nil {
			t.Fatalf("failed to seed tenant model: %v", err)
		}
	}

	providerName := "OpenAI"
	instanceName := "default"
	_, err := NewModelProviderService().getModelInstanceAndProviderByName(
		t.Context(),
		&providerName,
		&instanceName,
		&modelName,
		"user-1",
		&modelModule.APIConfig{},
	)
	if err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("getModelInstanceAndProviderByName() error = %v, want ambiguity error", err)
	}
}

func TestGetModelInstanceAndProviderByNameAllowsCustomTenantModel(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)

	custom := &entity.TenantModel{
		ID:         "custom-embedding-by-name",
		ProviderID: "provider-1",
		InstanceID: "instance-1",
		ModelName:  "tenant-only-embedding-by-name",
		ModelType:  int(entity.ModelTypeEmbedding),
		Status:     "active",
	}
	if err := db.Create(custom).Error; err != nil {
		t.Fatalf("failed to seed custom model: %v", err)
	}

	providerName := "OpenAI"
	instanceName := "default"
	modelName := custom.ModelName
	info, err := NewModelProviderService().getModelInstanceAndProviderByName(
		t.Context(),
		&providerName,
		&instanceName,
		&modelName,
		"user-1",
		&modelModule.APIConfig{},
	)
	if err != nil {
		t.Fatalf("getModelInstanceAndProviderByName() error = %v", err)
	}
	if info == nil || info.ModelEntity == nil || info.ModelEntity.ID != custom.ID {
		t.Fatalf("model entity = %#v, want %q", info, custom.ID)
	}
	if info.ModelInfo == nil || !info.ModelInfo.ModelTypeMap["embedding"] {
		t.Fatalf("model metadata = %#v, want embedding", info.ModelInfo)
	}
}

func TestGetModelInstanceAndProviderByIDKeepsTenantScope(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)

	foreign := &entity.TenantModelProvider{
		ID:           "provider-foreign",
		TenantID:     "tenant-foreign",
		ProviderName: "OpenAI",
	}
	instance := &entity.TenantModelInstance{
		ID:           "instance-foreign",
		ProviderID:   foreign.ID,
		InstanceName: "default",
		Status:       "active",
		Extra:        "{}",
	}
	model := &entity.TenantModel{
		ID:         "model-foreign",
		ProviderID: foreign.ID,
		InstanceID: instance.ID,
		ModelName:  "foreign-embedding",
		ModelType:  int(entity.ModelTypeEmbedding),
		Status:     "active",
	}
	for _, row := range []interface{}{foreign, instance, model} {
		if err := db.Create(row).Error; err != nil {
			t.Fatalf("failed to seed %T: %v", row, err)
		}
	}

	modelID := model.ID
	_, err := NewModelProviderService().getModelInstanceAndProviderByID(
		t.Context(),
		&modelID,
		"user-1",
		&modelModule.APIConfig{},
	)
	if err == nil || !errors.Is(err, errTenantModelNotFound) {
		t.Fatalf(
			"getModelInstanceAndProviderByID() error = %v, want scoped not-found error",
			err,
		)
	}
}

func TestEmbedTextClassifiesForeignModelAsNotFound(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)

	foreign := &entity.TenantModelProvider{
		ID:           "provider-foreign-public",
		TenantID:     "tenant-foreign",
		ProviderName: "OpenAI",
	}
	instance := &entity.TenantModelInstance{
		ID:           "instance-foreign-public",
		ProviderID:   foreign.ID,
		InstanceName: "default",
		Status:       "active",
		Extra:        "{}",
	}
	model := &entity.TenantModel{
		ID:         "model-foreign-public",
		ProviderID: foreign.ID,
		InstanceID: instance.ID,
		ModelName:  "foreign-embedding-public",
		ModelType:  int(entity.ModelTypeEmbedding),
		Status:     "active",
		Extra:      `{"max_dimension":1536,"max_batch_size":1,"dimensions":[1536]}`,
	}
	for _, row := range []interface{}{foreign, instance, model} {
		if err := db.Create(row).Error; err != nil {
			t.Fatalf("failed to seed %T: %v", row, err)
		}
	}

	modelID := model.ID
	response, code, err := NewModelProviderService().EmbedText(
		t.Context(), nil, nil, nil, &modelID, "user-1", []string{"test"},
		&modelModule.APIConfig{}, &modelModule.EmbeddingConfig{Dimension: 1536},
	)
	if err == nil || !errors.Is(err, errTenantModelNotFound) {
		t.Fatalf("EmbedText() error = %v, want scoped not-found error", err)
	}
	if code != common.CodeNotFound {
		t.Fatalf("code = %v, want %v", code, common.CodeNotFound)
	}
	if response != nil {
		t.Fatalf("response = %#v, want nil", response)
	}
}

func TestModelResolutionPreservesCanceledContext(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	providerName := "OpenAI"
	instanceName := "default"
	modelName := "gpt-test"
	modelID := "model-1"
	tests := []struct {
		name string
		call func() error
	}{
		{
			name: "by name",
			call: func() error {
				_, err := NewModelProviderService().getModelInstanceAndProviderByName(
					ctx, &providerName, &instanceName, &modelName, "user-1", nil,
				)
				return err
			},
		},
		{
			name: "by ID",
			call: func() error {
				_, err := NewModelProviderService().getModelInstanceAndProviderByID(
					ctx, &modelID, "user-1", nil,
				)
				return err
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.call()
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("error = %v, want context.Canceled identity", err)
			}
			if !errors.Is(err, errTenantModelLookup) {
				t.Fatalf("error = %v, want lookup-failure identity", err)
			}
		})
	}
}

func TestEmbedTextAllowsProviderModelWithoutEmbeddingLimits(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)

	custom := &entity.TenantModel{
		ID:         "custom-embedding-no-limits",
		ProviderID: "provider-1",
		InstanceID: "instance-1",
		ModelName:  "tenant-only-embedding-no-limits",
		ModelType:  int(entity.ModelTypeEmbedding),
		Status:     "active",
	}
	if err := db.Create(custom).Error; err != nil {
		t.Fatalf("failed to seed custom model: %v", err)
	}

	modelID := custom.ID
	_, code, err := NewModelProviderService().EmbedText(
		t.Context(),
		nil,
		nil,
		nil,
		&modelID,
		"user-1",
		[]string{"test"},
		&modelModule.APIConfig{},
		&modelModule.EmbeddingConfig{Dimension: 1024},
	)
	if code == common.CodeBadRequest && err != nil && strings.Contains(err.Error(), "embedding max dimension is nil") {
		t.Fatalf("EmbedText() rejected provider model with unknown limits: %v", err)
	}
}

func TestEmbedTextCallsCustomTenantModelProvider(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)

	previousAllowAnyHost := utility.AllowAnyHostForTest
	utility.AllowAnyHostForTest = true
	t.Cleanup(func() { utility.AllowAnyHostForTest = previousAllowAnyHost })

	var gotPath string
	var gotPathMu sync.Mutex
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPathMu.Lock()
		gotPath = r.URL.Path
		gotPathMu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.25,-0.5],"index":0}]}`))
	}))
	t.Cleanup(server.Close)

	if err := db.Model(&entity.TenantModelInstance{}).
		Where("id = ?", "instance-1").
		Updates(map[string]interface{}{"extra": `{"base_url":"` + server.URL + `"}`}).Error; err != nil {
		t.Fatalf("failed to set test provider URL: %v", err)
	}
	custom := &entity.TenantModel{
		ID:         "custom-embedding-provider-call",
		ProviderID: "provider-1",
		InstanceID: "instance-1",
		ModelName:  "tenant-only-embedding-provider-call",
		ModelType:  int(entity.ModelTypeEmbedding),
		Status:     "active",
		Extra:      `{"max_dimension":2,"max_batch_size":1,"dimensions":[2]}`,
	}
	if err := db.Create(custom).Error; err != nil {
		t.Fatalf("failed to seed custom model: %v", err)
	}

	modelID := custom.ID
	response, code, err := NewModelProviderService().EmbedText(
		t.Context(), nil, nil, nil, &modelID, "user-1", []string{"test"},
		&modelModule.APIConfig{}, &modelModule.EmbeddingConfig{Dimension: 2},
	)
	if err != nil {
		t.Fatalf("EmbedText() error = %v", err)
	}
	if code != common.CodeSuccess {
		t.Fatalf("code = %v, want %v", code, common.CodeSuccess)
	}
	gotPathMu.Lock()
	actualPath := gotPath
	gotPathMu.Unlock()
	if actualPath != "/embeddings" {
		t.Fatalf("provider path = %q, want /embeddings", actualPath)
	}
	if len(response) != 1 || len(response[0].Embedding) != 2 {
		t.Fatalf("response = %#v, want one 2-dimensional embedding", response)
	}
}

type nilRerankModel struct {
	modelModule.ModelDriver
}

func (m *nilRerankModel) Rerank(
	context.Context,
	*string,
	modelModule.RerankRequest,
	*modelModule.APIConfig,
	*modelModule.RerankConfig,
	*common.ModelUsage,
) (*modelModule.RerankResponse, error) {
	return nil, nil
}

func TestRerankDocumentRejectsNilProviderResponse(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	custom := &entity.TenantModel{
		ID:         "custom-rerank-model",
		ProviderID: "provider-1",
		InstanceID: "instance-1",
		ModelName:  "tenant-only-rerank-model",
		ModelType:  int(entity.ModelTypeRerank),
		Status:     "active",
	}
	if err := db.Create(custom).Error; err != nil {
		t.Fatalf("failed to seed custom model: %v", err)
	}

	service := NewModelProviderService()
	serviceDriver := &nilRerankModel{
		ModelDriver: modelModule.NewDummyModel(nil, modelModule.URLSuffix{}),
	}
	previousManager := dao.GetModelProviderManager()
	providerInfo := previousManager.FindProvider("OpenAI")
	if providerInfo == nil {
		t.Fatal("OpenAI provider metadata missing")
	}
	originalDriver := providerInfo.ModelDriver
	providerInfo.ModelDriver = serviceDriver
	t.Cleanup(func() { providerInfo.ModelDriver = originalDriver })

	modelID := custom.ID
	response, code, err := service.RerankDocument(
		t.Context(), nil, nil, nil, &modelID, "user-1", "query", []string{"doc"},
		&modelModule.APIConfig{}, &modelModule.RerankConfig{},
	)
	if err == nil || !strings.Contains(err.Error(), "empty rerank response") {
		t.Fatalf("RerankDocument() error = %v, want empty response error", err)
	}
	if code != common.CodeServerError {
		t.Fatalf("code = %v, want %v", code, common.CodeServerError)
	}
	if response != nil {
		t.Fatalf("response = %#v, want nil", response)
	}
}

func TestModelIDOnlyOperationsUseResolvedModelAndProviderNames(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	custom := &entity.TenantModel{
		ID:         "custom-media-model",
		ProviderID: "provider-1",
		InstanceID: "instance-1",
		ModelName:  "tenant-only-media-model",
		ModelType: int(
			entity.ModelTypeSpeech2Text |
				entity.ModelTypeTTS |
				entity.ModelTypeOCR,
		),
		Status: "active",
	}
	if err := db.Create(custom).Error; err != nil {
		t.Fatalf("failed to seed custom model: %v", err)
	}

	tests := []struct {
		name    string
		wantErr string
		call    func(*ModelProviderService, *string) (common.ErrorCode, error)
	}{
		{
			name:    "transcribe audio",
			wantErr: "file is missing",
			call: func(service *ModelProviderService, modelID *string) (common.ErrorCode, error) {
				_, code, err := service.TranscribeAudio(
					t.Context(), nil, nil, nil, modelID, "user-1", nil,
					&modelModule.APIConfig{}, nil,
				)
				return code, err
			},
		},
		{
			name:    "transcribe audio stream",
			wantErr: "file is missing",
			call: func(service *ModelProviderService, modelID *string) (common.ErrorCode, error) {
				return service.TranscribeAudioStream(
					t.Context(), nil, nil, nil, modelID, "user-1", nil,
					&modelModule.APIConfig{}, nil,
					func(*string, *string) error { return nil },
				)
			},
		},
		{
			name:    "audio speech",
			wantErr: "audio content is empty",
			call: func(service *ModelProviderService, modelID *string) (common.ErrorCode, error) {
				_, code, err := service.AudioSpeech(
					t.Context(), nil, nil, nil, modelID, "user-1", nil,
					&modelModule.APIConfig{}, nil,
				)
				return code, err
			},
		},
		{
			name:    "audio speech stream",
			wantErr: "audio content is empty",
			call: func(service *ModelProviderService, modelID *string) (common.ErrorCode, error) {
				return service.AudioSpeechStream(
					t.Context(), nil, nil, nil, modelID, "user-1", nil,
					&modelModule.APIConfig{}, nil,
					func(*string, *string) error { return nil },
				)
			},
		},
		{
			name:    "OCR file",
			wantErr: "no such method",
			call: func(service *ModelProviderService, modelID *string) (common.ErrorCode, error) {
				_, code, err := service.OCRFile(
					t.Context(), nil, nil, nil, modelID, "user-1", nil, nil,
					&modelModule.APIConfig{}, nil,
				)
				return code, err
			},
		},
		{
			name:    "parse file",
			wantErr: "no such method",
			call: func(service *ModelProviderService, modelID *string) (common.ErrorCode, error) {
				_, code, err := service.ParseFile(
					t.Context(), nil, nil, nil, modelID, "user-1", nil, nil,
					&modelModule.APIConfig{}, nil,
				)
				return code, err
			},
		},
	}

	service := NewModelProviderService()
	modelID := custom.ID
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			code, err := test.call(service, &modelID)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("operation error = %v, want %q", err, test.wantErr)
			}
			if code != common.CodeServerError {
				t.Fatalf("code = %v, want %v", code, common.CodeServerError)
			}
		})
	}
}

func TestEmbedTextRejectsInactiveAndWrongTypeCustomModels(t *testing.T) {
	tests := []struct {
		name      string
		status    string
		modelType entity.ModelType
		wantCode  common.ErrorCode
		wantErr   string
	}{
		{
			name:      "inactive embedding",
			status:    "inactive",
			modelType: entity.ModelTypeEmbedding,
			wantCode:  common.CodeServerError,
			wantErr:   "model is inactive",
		},
		{
			name:      "active chat",
			status:    "active",
			modelType: entity.ModelTypeChat,
			wantCode:  common.CodeNotFound,
			wantErr:   "is an embedding model",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			db := setupModelProviderServiceTestDB(t)
			useModelProviderServiceTestDB(t, db)
			seedModelProviderServiceScope(t, db)
			custom := &entity.TenantModel{
				ID:         "custom-model",
				ProviderID: "provider-1",
				InstanceID: "instance-1",
				ModelName:  "tenant-only-model",
				ModelType:  int(tt.modelType),
				Status:     tt.status,
				Extra:      `{"max_dimension":1024,"max_batch_size":8,"dimensions":[1024]}`,
			}
			if err := db.Create(custom).Error; err != nil {
				t.Fatalf("failed to seed custom model: %v", err)
			}

			modelID := custom.ID
			_, code, err := NewModelProviderService().EmbedText(
				t.Context(), nil, nil, nil, &modelID, "user-1", []string{"test"},
				&modelModule.APIConfig{}, &modelModule.EmbeddingConfig{Dimension: 1024},
			)
			if code != tt.wantCode {
				t.Fatalf("code = %v, want %v", code, tt.wantCode)
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("EmbedText() error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestModelProviderServiceResolveModelContextLength(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	// Seed a tenant chat model that maps to a real factory-catalog model
	// (Anthropic / claude-opus-4-8 has content_length=1000000, max_output=128000).
	activeStatus := "1"
	rows := []interface{}{
		&entity.UserTenant{ID: "user-tenant-cl", UserID: "user-1", TenantID: "tenant-cl", Role: "owner", InvitedBy: "user-1", Status: &activeStatus},
		&entity.TenantModelProvider{ID: "provider-anthropic", TenantID: "tenant-cl", ProviderName: "Anthropic"},
		&entity.TenantModelInstance{ID: "instance-anthropic", ProviderID: "provider-anthropic", InstanceName: "default", APIKey: "sk-anthropic", Status: "active", Extra: "{}"},
		&entity.TenantModel{ID: "model-claude", ProviderID: "provider-anthropic", InstanceID: "instance-anthropic", ModelName: "claude-opus-4-8", ModelType: int(entity.ModelTypeChat), Status: "active"},
	}
	for _, row := range rows {
		if err := db.Create(row).Error; err != nil {
			t.Fatalf("failed to seed %T: %v", row, err)
		}
	}

	svc := NewModelProviderService()
	ctx := t.Context()

	// UUID path: resolves content_length (context window) from the factory
	// catalog, NOT max_output.
	got, err := svc.ResolveModelContextLength(ctx, "user-1", "model-claude")
	if err != nil {
		t.Fatalf("ResolveModelContextLength(uuid) error = %v", err)
	}
	if got != 1000000 {
		t.Fatalf("uuid content_length = %d, want 1000000 (must be the context window, not max_output=128000)", got)
	}

	// Composite "model@instance@provider" path resolves the same value.
	got2, err := svc.ResolveModelContextLength(ctx, "user-1", "claude-opus-4-8@default@Anthropic")
	if err != nil {
		t.Fatalf("ResolveModelContextLength(composite) error = %v", err)
	}
	if got2 != 1000000 {
		t.Fatalf("composite content_length = %d, want 1000000", got2)
	}
}

func TestModelProviderServiceResolveModelContextLengthUnknownModel(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)

	// A model that does not exist in the factory catalog resolves to 0 so the
	// caller falls back to its default context length instead of failing.
	got, err := NewModelProviderService().ResolveModelContextLength(
		t.Context(), "user-1", "gpt-no-such-model@default@OpenAI")
	if err != nil {
		t.Fatalf("ResolveModelContextLength(unknown) error = %v", err)
	}
	if got != 0 {
		t.Fatalf("unknown model content_length = %d, want 0", got)
	}
}

// TestModelProviderServiceResolveModelContextLengthOverride verifies that the
// tenant-configured "max_tokens" override in tenant_model.extra wins over the
// catalog content_length through the service delegation. UUID resolution is
// unscoped (globally unique); the composite path needs the real tenant id to
// locate the tenant's provider/instance/model rows.
func TestModelProviderServiceResolveModelContextLengthOverride(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	activeStatus := "1"
	rows := []interface{}{
		&entity.UserTenant{ID: "user-tenant-cl", UserID: "user-1", TenantID: "tenant-cl", Role: "owner", InvitedBy: "user-1", Status: &activeStatus},
		&entity.TenantModelProvider{ID: "provider-anthropic", TenantID: "tenant-cl", ProviderName: "Anthropic"},
		&entity.TenantModelInstance{ID: "instance-anthropic", ProviderID: "provider-anthropic", InstanceName: "default", APIKey: "sk-anthropic", Status: "active", Extra: "{}"},
		&entity.TenantModel{ID: "model-claude", ProviderID: "provider-anthropic", InstanceID: "instance-anthropic", ModelName: "claude-opus-4-8", ModelType: int(entity.ModelTypeChat), Status: "active", Extra: `{"max_tokens": 4096}`},
	}
	for _, row := range rows {
		if err := db.Create(row).Error; err != nil {
			t.Fatalf("failed to seed %T: %v", row, err)
		}
	}

	svc := NewModelProviderService()
	ctx := t.Context()

	// UUID path: the 4096 override wins over catalog content_length 1000000.
	got, err := svc.ResolveModelContextLength(ctx, "user-1", "model-claude")
	if err != nil {
		t.Fatalf("ResolveModelContextLength(override uuid) error = %v", err)
	}
	if got != 4096 {
		t.Fatalf("uuid override content_length = %d, want 4096 (custom override, not catalog 1000000)", got)
	}

	// Composite path with the real tenant id honors the same override.
	got2, err := svc.ResolveModelContextLength(ctx, "tenant-cl", "claude-opus-4-8@default@Anthropic")
	if err != nil {
		t.Fatalf("ResolveModelContextLength(override composite) error = %v", err)
	}
	if got2 != 4096 {
		t.Fatalf("composite override content_length = %d, want 4096", got2)
	}
}

func TestModelProviderServiceAlterModelRejectsInvalidStatus(t *testing.T) {
	ctx := t.Context()
	code, err := NewModelProviderService().AlterModel(ctx, "OpenAI", "default", "", "user-1", "model-1", map[string]interface{}{"status": "disabled"})
	if err == nil {
		t.Fatalf("AlterModel() error = nil, want invalid status error")
	}
	if code != common.CodeBadRequest {
		t.Fatalf("code = %v, want %v", code, common.CodeBadRequest)
	}
	if !strings.Contains(err.Error(), "status must be") {
		t.Fatalf("error = %v, want status validation message", err)
	}
}

func TestModelProviderServiceAlterModelRejectsMissingModelSelector(t *testing.T) {
	ctx := t.Context()
	code, err := NewModelProviderService().AlterModel(ctx, "OpenAI", "default", "", "user-1", "", map[string]interface{}{"status": "active"})
	if err == nil {
		t.Fatalf("AlterModel() error = nil, want missing model selector error")
	}
	if code != common.CodeBadRequest {
		t.Fatalf("code = %v, want %v", code, common.CodeBadRequest)
	}
	if !strings.Contains(err.Error(), "model name or model ID is required") {
		t.Fatalf("error = %v, want missing model selector message", err)
	}
}

func TestModelProviderServiceAlterModelRejectsWrongScopedModelID(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	useModelProviderServiceTestDB(t, db)
	seedModelProviderServiceScope(t, db)
	if err := db.Create(&entity.TenantModelInstance{ID: "instance-2", ProviderID: "provider-1", InstanceName: "other", APIKey: "sk-test", Status: "active", Extra: "{}"}).Error; err != nil {
		t.Fatalf("failed to seed second instance: %v", err)
	}

	ctx := t.Context()
	code, err := NewModelProviderService().AlterModel(ctx, "OpenAI", "other", "", "user-1", "model-1", map[string]interface{}{"status": "inactive"})
	if err == nil {
		t.Fatalf("AlterModel() error = nil, want not found error")
	}
	if code != common.CodeNotFound {
		t.Fatalf("code = %v, want %v", code, common.CodeNotFound)
	}
}

func TestReconcileNvidiaInstanceModelsRejectsDuplicateExistingNamesWithoutMutation(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	provider := &entity.TenantModelProvider{
		ID:           "provider-nvidia",
		TenantID:     "tenant-1",
		ProviderName: "NVIDIA",
	}
	instance := &entity.TenantModelInstance{
		ID:           "instance-nvidia",
		ProviderID:   provider.ID,
		InstanceName: "default",
		Status:       "active",
		Extra:        "{}",
	}
	duplicates := []*entity.TenantModel{
		{
			ID:         "duplicate-a",
			ProviderID: provider.ID,
			InstanceID: instance.ID,
			ModelName:  "nvidia/duplicate",
			ModelType:  int(entity.ModelTypeChat),
			Status:     "active",
			Extra:      `{"slot":"a"}`,
		},
		{
			ID:         "duplicate-b",
			ProviderID: provider.ID,
			InstanceID: instance.ID,
			ModelName:  "nvidia/duplicate",
			ModelType:  int(entity.ModelTypeEmbedding),
			Status:     "inactive",
			Extra:      `{"slot":"b"}`,
		},
	}
	for _, row := range []interface{}{provider, instance, duplicates[0], duplicates[1]} {
		if err := db.Create(row).Error; err != nil {
			t.Fatalf("seed %T: %v", row, err)
		}
	}

	err := NewModelProviderService().reconcileNvidiaInstanceModels(
		t.Context(),
		db,
		provider,
		instance,
		[]modelModule.ListModelResponse{{
			Name:       "nvidia/duplicate",
			ModelTypes: []string{"chat", "vision"},
		}},
	)
	if err == nil || !errors.Is(err, errDuplicateExistingModel) {
		t.Fatalf(
			"reconcileNvidiaInstanceModels() error = %v, want duplicate existing model error",
			err,
		)
	}

	var got []entity.TenantModel
	if err := db.Where("provider_id = ? AND instance_id = ?", provider.ID, instance.ID).
		Order("id").Find(&got).Error; err != nil {
		t.Fatalf("reload models: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("model count = %d, want 2", len(got))
	}
	if got[0].ID != duplicates[0].ID ||
		got[0].ModelType != duplicates[0].ModelType ||
		got[0].Status != duplicates[0].Status ||
		got[0].Extra != duplicates[0].Extra ||
		got[1].ID != duplicates[1].ID ||
		got[1].ModelType != duplicates[1].ModelType ||
		got[1].Status != duplicates[1].Status ||
		got[1].Extra != duplicates[1].Extra {
		t.Fatalf("models after duplicate failure = %#v, want original rows", got)
	}
}

func TestReconcileNvidiaInstanceModelsAddsUpdatesAndPreservesOmitted(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	provider := &entity.TenantModelProvider{ID: "provider-nvidia", TenantID: "tenant-1", ProviderName: "NVIDIA"}
	instance := &entity.TenantModelInstance{ID: "instance-nvidia", ProviderID: provider.ID, InstanceName: "default", APIKey: "nvapi-test", Status: "active", Extra: "{}"}
	rows := []interface{}{
		provider,
		instance,
		&entity.TenantModel{
			ID:         "keep-id",
			ProviderID: provider.ID,
			InstanceID: instance.ID,
			ModelName:  "nvidia/keep",
			ModelType:  int(entity.ModelTypeChat),
			Status:     "inactive",
			Extra:      `{"max_tokens":4096,"verify":"success","custom":"preserved"}`,
		},
		&entity.TenantModel{
			ID:         "stale-id",
			ProviderID: provider.ID,
			InstanceID: instance.ID,
			ModelName:  "nvidia/stale",
			ModelType:  int(entity.ModelTypeChat),
			Status:     "active",
			Extra:      `{}`,
		},
	}
	for _, row := range rows {
		if err := db.Create(row).Error; err != nil {
			t.Fatalf("seed %T: %v", row, err)
		}
	}

	maxTokens := 131072
	maxDimension := 2048
	remote := []modelModule.ListModelResponse{
		{Name: "nvidia/keep", MaxOutput: &maxTokens, ModelTypes: []string{"chat", "vision"}},
		{Name: "nvidia/new-embed", MaxOutput: ptrService(8192), MaxDimension: &maxDimension, Dimensions: []int{1024, 2048}, ModelTypes: []string{"embedding"}},
	}

	err := NewModelProviderService().reconcileNvidiaInstanceModels(context.Background(), db, provider, instance, remote)
	if err != nil {
		t.Fatalf("reconcileNvidiaInstanceModels() error = %v", err)
	}

	var got []*entity.TenantModel
	if err = db.Order("model_name").Find(&got).Error; err != nil {
		t.Fatalf("list models: %v", err)
	}
	if len(got) != 3 ||
		got[0].ModelName != "nvidia/keep" ||
		got[1].ModelName != "nvidia/new-embed" ||
		got[2].ModelName != "nvidia/stale" {
		t.Fatalf("models = %#v, want keep, new, and preserved stale", got)
	}
	if got[0].ID != "keep-id" || got[0].Status != "inactive" {
		t.Fatalf("retained model identity/status = %q/%q", got[0].ID, got[0].Status)
	}
	if got[0].ModelType != int(entity.ModelTypeChat|entity.ModelTypeImage2Text) {
		t.Fatalf("retained model type = %d", got[0].ModelType)
	}
	var keepExtra map[string]interface{}
	if err := json.Unmarshal([]byte(got[0].Extra), &keepExtra); err != nil {
		t.Fatalf("decode retained extra: %v", err)
	}
	if keepExtra["custom"] != "preserved" || keepExtra["verify"] != "success" || int(keepExtra["max_tokens"].(float64)) != maxTokens {
		t.Fatalf("retained extra = %#v", keepExtra)
	}
	var newExtra map[string]interface{}
	if err := json.Unmarshal([]byte(got[1].Extra), &newExtra); err != nil {
		t.Fatalf("decode new extra: %v", err)
	}
	if newExtra["verify"] != entity.ModelVerifyUnknown || int(newExtra["max_dimension"].(float64)) != maxDimension {
		t.Fatalf("new extra = %#v", newExtra)
	}
}

func TestReconcileNvidiaInstanceModelsSkipsUnchangedUpdates(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	provider := &entity.TenantModelProvider{
		ID:           "provider-nvidia",
		TenantID:     "tenant-1",
		ProviderName: "NVIDIA",
	}
	instance := &entity.TenantModelInstance{
		ID:           "instance-nvidia",
		ProviderID:   provider.ID,
		InstanceName: "default",
		Status:       "active",
		Extra:        "{}",
	}
	existing := &entity.TenantModel{
		ID:         "existing-id",
		ProviderID: provider.ID,
		InstanceID: instance.ID,
		ModelName:  "nvidia/existing",
		ModelType:  int(entity.ModelTypeEmbedding),
		Status:     "active",
		Extra:      `{}`,
	}
	for _, row := range []interface{}{provider, instance, existing} {
		if err := db.Create(row).Error; err != nil {
			t.Fatalf("seed %T: %v", row, err)
		}
	}
	if err := db.Exec(`
		CREATE TRIGGER ignore_unchanged_nvidia_update
		BEFORE UPDATE ON tenant_model
		WHEN NEW.model_type = OLD.model_type AND NEW.extra = OLD.extra
		BEGIN
			SELECT RAISE(IGNORE);
		END
	`).Error; err != nil {
		t.Fatalf("create unchanged-update trigger: %v", err)
	}

	maxDimension := 2048
	remote := []modelModule.ListModelResponse{{
		Name:         existing.ModelName,
		MaxOutput:    ptrService(8192),
		MaxDimension: &maxDimension,
		Dimensions:   []int{1024, 2048},
		ModelTypes:   []string{"embedding"},
	}}
	service := NewModelProviderService()
	if err := service.reconcileNvidiaInstanceModels(
		t.Context(), db, provider, instance, remote,
	); err != nil {
		t.Fatalf("first reconcileNvidiaInstanceModels() error = %v", err)
	}
	if err := service.reconcileNvidiaInstanceModels(
		t.Context(), db, provider, instance, remote,
	); err != nil {
		t.Fatalf("second reconcileNvidiaInstanceModels() error = %v", err)
	}
}

func TestReconcileNvidiaInstanceModelsPreservesOmittedModels(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	provider := &entity.TenantModelProvider{ID: "provider-nvidia", TenantID: "tenant-1", ProviderName: "NVIDIA"}
	instance := &entity.TenantModelInstance{ID: "instance-nvidia", ProviderID: provider.ID, InstanceName: "default", Status: "active", Extra: "{}"}
	omitted := &entity.TenantModel{
		ID:         "omitted-id",
		ProviderID: provider.ID,
		InstanceID: instance.ID,
		ModelName:  "nvidia/temporarily-omitted",
		ModelType:  int(entity.ModelTypeChat),
		Status:     "active",
		Extra:      "{}",
	}
	for _, row := range []interface{}{provider, instance, omitted} {
		if err := db.Create(row).Error; err != nil {
			t.Fatalf("seed %T: %v", row, err)
		}
	}

	remote := []modelModule.ListModelResponse{{Name: "nvidia/current", ModelTypes: []string{"chat"}}}
	if err := NewModelProviderService().reconcileNvidiaInstanceModels(t.Context(), db, provider, instance, remote); err != nil {
		t.Fatalf("reconcileNvidiaInstanceModels() error = %v", err)
	}

	var got entity.TenantModel
	if err := db.Where("id = ?", omitted.ID).First(&got).Error; err != nil {
		t.Fatalf("omitted model was deleted: %v", err)
	}
}

func TestReconcileNvidiaInstanceModelsRejectsEmptyDiscoveryWithoutMutation(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	provider := &entity.TenantModelProvider{ID: "provider-nvidia", TenantID: "tenant-1", ProviderName: "NVIDIA"}
	instance := &entity.TenantModelInstance{ID: "instance-nvidia", ProviderID: provider.ID, InstanceName: "default", Status: "active", Extra: "{}"}
	existing := &entity.TenantModel{ID: "keep-id", ProviderID: provider.ID, InstanceID: instance.ID, ModelName: "nvidia/keep", ModelType: int(entity.ModelTypeChat), Status: "active", Extra: "{}"}
	for _, row := range []interface{}{provider, instance, existing} {
		if err := db.Create(row).Error; err != nil {
			t.Fatalf("seed %T: %v", row, err)
		}
	}

	err := NewModelProviderService().reconcileNvidiaInstanceModels(context.Background(), db, provider, instance, nil)
	if err == nil {
		t.Fatal("reconcileNvidiaInstanceModels() error = nil, want empty discovery error")
	}
	var count int64
	if err := db.Model(&entity.TenantModel{}).Where("id = ?", existing.ID).Count(&count).Error; err != nil {
		t.Fatalf("count retained model: %v", err)
	}
	if count != 1 {
		t.Fatalf("retained model count = %d, want 1", count)
	}
}

func TestReconcileNvidiaInstanceModelsRollsBackPartialRefresh(t *testing.T) {
	db := setupModelProviderServiceTestDB(t)
	provider := &entity.TenantModelProvider{ID: "provider-nvidia", TenantID: "tenant-1", ProviderName: "NVIDIA"}
	instance := &entity.TenantModelInstance{ID: "instance-nvidia", ProviderID: provider.ID, InstanceName: "default", Status: "active", Extra: "{}"}
	existing := &entity.TenantModel{
		ID:         "keep-id",
		ProviderID: provider.ID,
		InstanceID: instance.ID,
		ModelName:  "nvidia/keep",
		ModelType:  int(entity.ModelTypeChat),
		Status:     "active",
		Extra:      "{invalid-json",
	}
	for _, row := range []interface{}{provider, instance, existing} {
		if err := db.Create(row).Error; err != nil {
			t.Fatalf("seed %T: %v", row, err)
		}
	}

	remote := []modelModule.ListModelResponse{
		{Name: "nvidia/new", ModelTypes: []string{"chat"}},
		{Name: "nvidia/keep", ModelTypes: []string{"chat"}},
	}
	err := NewModelProviderService().reconcileNvidiaInstanceModels(context.Background(), db, provider, instance, remote)
	if err == nil {
		t.Fatal("reconcileNvidiaInstanceModels() error = nil, want metadata error")
	}

	var got []*entity.TenantModel
	if err := db.Order("model_name").Find(&got).Error; err != nil {
		t.Fatalf("list models: %v", err)
	}
	if len(got) != 1 || got[0].ID != existing.ID {
		t.Fatalf("models after rollback = %#v, want only original model", got)
	}
}

func ptrService[T any](value T) *T {
	return &value
}

func TestParseModelName(t *testing.T) {
	tests := []struct {
		name         string
		composite    string
		wantModel    string
		wantInstance string
		wantProvider string
		wantErr      bool
	}{
		{
			name:         "three parts: model@instance@provider",
			composite:    "text-embedding-3-small@primary@OpenAI",
			wantModel:    "text-embedding-3-small",
			wantInstance: "primary",
			wantProvider: "OpenAI",
		},
		{
			name:         "two parts: model@provider defaults instance",
			composite:    "BAAI/bge-m3@Builtin",
			wantModel:    "BAAI/bge-m3",
			wantInstance: "default",
			wantProvider: "Builtin",
		},
		{
			name:      "single part bare name returns error",
			composite: "BAAI/bge-m3",
			wantErr:   true,
		},
		{
			name:         "embedded @ in modelName preserved (four parts)",
			composite:    "text-embedding-nomic-embed-text-v1.5@q8_0@default@LM-Studio",
			wantModel:    "text-embedding-nomic-embed-text-v1.5@q8_0",
			wantInstance: "default",
			wantProvider: "LM-Studio",
		},
		{
			name:         "multiple embedded @ in modelName preserved (five parts)",
			composite:    "org/repo@tag@1.0@default@Ollama",
			wantModel:    "org/repo@tag@1.0",
			wantInstance: "default",
			wantProvider: "Ollama",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, instance, provider, err := parseModelName(tt.composite)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("parseModelName(%q) error = nil, want error", tt.composite)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseModelName(%q) unexpected error: %v", tt.composite, err)
			}
			if model != tt.wantModel {
				t.Errorf("parseModelName(%q) model = %q, want %q", tt.composite, model, tt.wantModel)
			}
			if instance != tt.wantInstance {
				t.Errorf("parseModelName(%q) instance = %q, want %q", tt.composite, instance, tt.wantInstance)
			}
			if provider != tt.wantProvider {
				t.Errorf("parseModelName(%q) provider = %q, want %q", tt.composite, provider, tt.wantProvider)
			}
		})
	}
}

func TestSplitRightAnchoredModelName(t *testing.T) {
	tests := []struct {
		name         string
		composite    string
		wantModel    string
		wantInstance string
		wantProvider string
	}{
		{
			name:         "three parts: model@instance@provider",
			composite:    "text-embedding-3-small@primary@OpenAI",
			wantModel:    "text-embedding-3-small",
			wantInstance: "primary",
			wantProvider: "OpenAI",
		},
		{
			name:         "two parts: model@provider defaults instance",
			composite:    "BAAI/bge-m3@Builtin",
			wantModel:    "BAAI/bge-m3",
			wantInstance: "default",
			wantProvider: "Builtin",
		},
		{
			name:         "single part bare name returns empty provider and instance",
			composite:    "BAAI/bge-m3",
			wantModel:    "BAAI/bge-m3",
			wantInstance: "",
			wantProvider: "",
		},
		{
			// Regression for the CodeRabbit "Major" comment on PR #16468:
			// a 2-segment key whose '@' is part of the model name (not a
			// provider separator) must stay bare. Without this branch the
			// helper would return ("text-embedding-nomic-embed-text-v1.5",
			// "default", "q8_0"), mis-classifying the quantization tag as a
			// provider and missing the TEI fast path's `modelName == teiModel`
			// match when TEI_MODEL is the full embedded string.
			name:         "two parts bare default with embedded '@' stays bare",
			composite:    "text-embedding-nomic-embed-text-v1.5@q8_0",
			wantModel:    "text-embedding-nomic-embed-text-v1.5@q8_0",
			wantInstance: "",
			wantProvider: "",
		},
		{
			name:         "embedded @ in modelName preserved (four parts)",
			composite:    "text-embedding-nomic-embed-text-v1.5@q8_0@default@LM-Studio",
			wantModel:    "text-embedding-nomic-embed-text-v1.5@q8_0",
			wantInstance: "default",
			wantProvider: "LM-Studio",
		},
		{
			name:         "multiple embedded @ in modelName preserved (five parts)",
			composite:    "org/repo@tag@1.0@default@Ollama",
			wantModel:    "org/repo@tag@1.0",
			wantInstance: "default",
			wantProvider: "Ollama",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, instance, provider := splitRightAnchoredModelName(tt.composite)
			if model != tt.wantModel {
				t.Errorf("splitRightAnchoredModelName(%q) model = %q, want %q", tt.composite, model, tt.wantModel)
			}
			if instance != tt.wantInstance {
				t.Errorf("splitRightAnchoredModelName(%q) instance = %q, want %q", tt.composite, instance, tt.wantInstance)
			}
			if provider != tt.wantProvider {
				t.Errorf("splitRightAnchoredModelName(%q) provider = %q, want %q", tt.composite, provider, tt.wantProvider)
			}
		})
	}
}
