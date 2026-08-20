package server

import (
	"reflect"
	"testing"

	"github.com/spf13/viper"
)

func TestSafeConfigLogMetadataDoesNotExposeValues(t *testing.T) {
	v := viper.New()
	v.Set("mysql", map[string]interface{}{
		"host":     "mysql",
		"password": "do-not-log",
	})
	v.Set("redis", map[string]interface{}{
		"host":     "redis",
		"password": "also-do-not-log",
	})

	got := safeConfigLogMetadata(v)
	want := []string{"mysql", "redis"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("safeConfigLogMetadata() = %#v, want %#v", got, want)
	}
}
