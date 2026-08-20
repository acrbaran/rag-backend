//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//

package nats

import "testing"

func TestIngestorControlSubjectRejectsSubjectInjection(t *testing.T) {
	for _, value := range []string{"", "worker.one", "worker one", "worker*", "worker>"} {
		if _, err := ingestorControlSubject(value); err == nil {
			t.Fatalf("ingestorControlSubject(%q) accepted invalid id", value)
		}
	}
	subject, err := ingestorControlSubject("ingestor-worker_01")
	if err != nil || subject != "control.ingestor.ingestor-worker_01.shutdown" {
		t.Fatalf("subject = %q, err = %v", subject, err)
	}
}
