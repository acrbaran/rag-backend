//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//

package nats

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"time"

	gonats "github.com/nats-io/nats.go"
)

const ingestorControlPrefix = "control.ingestor"

var ingestorControlID = regexp.MustCompile(`^[A-Za-z0-9_-]{1,128}$`)

type ingestorShutdownCommand struct {
	TaskID     string `json:"task_id"`
	IngestorID string `json:"ingestor_id"`
}

func ingestorControlSubject(ingestorID string) (string, error) {
	if !ingestorControlID.MatchString(ingestorID) {
		return "", fmt.Errorf("invalid ingestor id")
	}
	return fmt.Sprintf("%s.%s.shutdown", ingestorControlPrefix, ingestorID), nil
}

// PublishIngestorShutdown sends a targeted, non-persistent control message.
// The admin handler first requires a fresh heartbeat, so publishing to a stale
// or unknown process is rejected before this method is called.
func (n *NatsEngine) PublishIngestorShutdown(ingestorID, taskID string) error {
	if n == nil || n.nc == nil {
		return fmt.Errorf("NATS connection is not initialized")
	}
	subject, err := ingestorControlSubject(ingestorID)
	if err != nil {
		return err
	}
	payload, err := json.Marshal(ingestorShutdownCommand{TaskID: taskID, IngestorID: ingestorID})
	if err != nil {
		return err
	}
	if err = n.nc.Publish(subject, payload); err != nil {
		return fmt.Errorf("publish ingestor shutdown: %w", err)
	}
	if err = n.nc.FlushTimeout(2 * time.Second); err != nil {
		return fmt.Errorf("flush ingestor shutdown: %w", err)
	}
	return nil
}

// SubscribeIngestorShutdown receives commands targeted to this exact ingestor.
// Only the opaque task id is surfaced; malformed or cross-target messages are
// ignored without logging their body.
func (n *NatsEngine) SubscribeIngestorShutdown(ctx context.Context, ingestorID string) (<-chan string, error) {
	if n == nil || n.nc == nil {
		return nil, fmt.Errorf("NATS connection is not initialized")
	}
	subject, err := ingestorControlSubject(ingestorID)
	if err != nil {
		return nil, err
	}
	commands := make(chan string, 1)
	subscription, err := n.nc.Subscribe(subject, func(msg *gonats.Msg) {
		var command ingestorShutdownCommand
		if json.Unmarshal(msg.Data, &command) != nil || command.IngestorID != ingestorID || command.TaskID == "" {
			return
		}
		select {
		case commands <- command.TaskID:
		default:
		}
	})
	if err != nil {
		return nil, fmt.Errorf("subscribe ingestor shutdown: %w", err)
	}
	go func() {
		<-ctx.Done()
		_ = subscription.Unsubscribe()
	}()
	return commands, nil
}
