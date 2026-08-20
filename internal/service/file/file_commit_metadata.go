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

package file

type fileMetadataChange struct {
	Operation   string
	OldName     string
	NewName     string
	OldParentID string
	NewParentID string
}

func metadataDiffEntries(
	committedEntry map[string]interface{},
	liveName string,
	liveParentID string,
) []fileMetadataChange {
	changes := make([]fileMetadataChange, 0, 2)
	committedName, _ := committedEntry["name"].(string)
	committedParentID, _ := committedEntry["parent_id"].(string)

	if liveName != committedName {
		changes = append(changes, fileMetadataChange{
			Operation:   "rename",
			OldName:     committedName,
			NewName:     liveName,
			OldParentID: committedParentID,
			NewParentID: liveParentID,
		})
	}

	if liveParentID != committedParentID {
		changes = append(changes, fileMetadataChange{
			Operation:   "move",
			OldParentID: committedParentID,
			NewParentID: liveParentID,
		})
	}

	return changes
}
