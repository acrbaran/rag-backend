package file

import "testing"

func TestMetadataDiffEntriesDetectRenameAndMove(t *testing.T) {
	changes := metadataDiffEntries(map[string]interface{}{
		"name": "old.txt", "parent_id": "folder-old",
	}, "new.txt", "folder-new")

	if len(changes) != 2 {
		t.Fatalf("expected rename and move, got %#v", changes)
	}
	if changes[0].Operation != "rename" || changes[0].OldName != "old.txt" || changes[0].NewName != "new.txt" {
		t.Fatalf("unexpected rename change: %#v", changes[0])
	}
	if changes[0].OldParentID != "folder-old" || changes[0].NewParentID != "folder-new" {
		t.Fatalf("unexpected rename parent metadata: %#v", changes[0])
	}
	if changes[1].Operation != "move" || changes[1].OldParentID != "folder-old" || changes[1].NewParentID != "folder-new" {
		t.Fatalf("unexpected move change: %#v", changes[1])
	}
}

func TestMetadataDiffEntriesIgnoresUnchangedMetadata(t *testing.T) {
	changes := metadataDiffEntries(map[string]interface{}{
		"name": "same.txt", "parent_id": "folder",
	}, "same.txt", "folder")

	if len(changes) != 0 {
		t.Fatalf("expected no metadata changes, got %#v", changes)
	}
}
