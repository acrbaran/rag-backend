//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//

package admin

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"ragflow/internal/common"
)

type recordingIngestorShutdownPublisher struct {
	ingestorID string
	taskID     string
	err        error
}

func (p *recordingIngestorShutdownPublisher) PublishIngestorShutdown(ingestorID, taskID string) error {
	p.ingestorID = ingestorID
	p.taskID = taskID
	return p.err
}

func TestShutdownIngestorSubmitsTargetedControlMessage(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &ServerStore{servers: map[string]*common.BaseMessage{
		"ingestor-worker-1": {
			ServerName: "ingestor-worker-1",
			ServerType: common.ServerTypeIngestion,
			Timestamp:  time.Now(),
		},
	}}
	publisher := &recordingIngestorShutdownPublisher{}
	h := &Handler{serverStore: store, ingestorShutdownPublisher: publisher}
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodDelete, "/api/v1/admin/ingestors", bytes.NewBufferString(`{"ingestor_name":"ingestor-worker-1"}`))
	c.Request.Header.Set("Content-Type", "application/json")

	h.ShutdownIngestor(c)

	if recorder.Code != http.StatusAccepted {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	if publisher.ingestorID != "ingestor-worker-1" || publisher.taskID == "" {
		t.Fatalf("publisher called with ingestor=%q task=%q", publisher.ingestorID, publisher.taskID)
	}
}

func TestShutdownIngestorRejectsUnknownTarget(t *testing.T) {
	gin.SetMode(gin.TestMode)
	publisher := &recordingIngestorShutdownPublisher{}
	h := &Handler{serverStore: &ServerStore{servers: map[string]*common.BaseMessage{}}, ingestorShutdownPublisher: publisher}
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodDelete, "/api/v1/admin/ingestors", bytes.NewBufferString(`{"ingestor_name":"unknown"}`))
	c.Request.Header.Set("Content-Type", "application/json")

	h.ShutdownIngestor(c)

	if publisher.taskID != "" {
		t.Fatal("unknown ingestor must not publish a shutdown command")
	}
}
