package service

import (
	"strings"
)

// normalizeListedModelName keeps exact remote model IDs unchanged and repairs
// a legacy routing suffix only when the provider's live catalog proves the
// suffix-free ID exists. Model IDs may legitimately contain '@', so no suffix
// is removed without that positive catalog match.
func normalizeListedModelName(requested string, listedNames []string) (string, bool) {
	requested = strings.TrimSpace(requested)
	if requested == "" || len(listedNames) == 0 {
		return requested, false
	}

	remoteNames := make(map[string]struct{}, len(listedNames))
	for _, listedName := range listedNames {
		name := strings.TrimSpace(listedName)
		if name != "" {
			remoteNames[name] = struct{}{}
		}
	}
	if _, ok := remoteNames[requested]; ok {
		return requested, false
	}

	if candidate, ok := strings.CutSuffix(requested, "@openai"); ok {
		candidate = strings.TrimSpace(candidate)
		if _, exists := remoteNames[candidate]; exists {
			return candidate, true
		}
	}

	return requested, false
}
