package chunk

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"testing"
)

func TestRetrievalTestLogsDoNotUseQuestionContent(t *testing.T) {
	function, fileSet := parseRetrievalTestSource(t)

	ast.Inspect(function.Body, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok || !isCommonLogCall(call) {
			return true
		}
		for _, argument := range call.Args {
			if usesQuestionContent(argument) {
				t.Errorf("RetrievalTest log at %s uses raw question content", fileSet.Position(call.Pos()))
				break
			}
		}
		return true
	})
}

func TestRetrievalTestLogsSafeQuestionMetadata(t *testing.T) {
	function, _ := parseRetrievalTestSource(t)
	requiredKeys := map[string]bool{
		"question_length":          false,
		"original_question_length": false,
		"modified_question_length": false,
		"transformed":              false,
		"chunk_count":              false,
	}

	ast.Inspect(function.Body, func(node ast.Node) bool {
		literal, ok := node.(*ast.BasicLit)
		if !ok || literal.Kind != token.STRING {
			return true
		}
		for key := range requiredKeys {
			if literal.Value == `"`+key+`"` {
				requiredKeys[key] = true
			}
		}
		return true
	})

	for key, found := range requiredKeys {
		if !found {
			t.Errorf("RetrievalTest logs missing safe metadata key %q", key)
		}
	}
}

func parseRetrievalTestSource(t *testing.T) (*ast.FuncDecl, *token.FileSet) {
	t.Helper()
	sourcePath := os.Getenv("RAGFLOW_CHUNK_SOURCE")
	if sourcePath == "" {
		sourcePath = "chunk.go"
	}
	fileSet := token.NewFileSet()
	file, err := parser.ParseFile(fileSet, sourcePath, nil, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", sourcePath, err)
	}
	for _, declaration := range file.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if ok && function.Name.Name == "RetrievalTest" {
			return function, fileSet
		}
	}
	t.Fatalf("RetrievalTest not found in %s", sourcePath)
	return nil, nil
}

func isCommonLogCall(call *ast.CallExpr) bool {
	selector, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	packageName, ok := selector.X.(*ast.Ident)
	if !ok || packageName.Name != "common" {
		return false
	}
	switch selector.Sel.Name {
	case "Debug", "Info", "Warn", "Error":
		return true
	default:
		return false
	}
}

func usesQuestionContent(expression ast.Expr) bool {
	found := false
	ast.Inspect(expression, func(node ast.Node) bool {
		if found {
			return false
		}
		switch value := node.(type) {
		case *ast.CallExpr:
			function, ok := value.Fun.(*ast.Ident)
			if ok && function.Name == "len" {
				return false
			}
		case *ast.SelectorExpr:
			identifier, ok := value.X.(*ast.Ident)
			if ok && identifier.Name == "req" && value.Sel.Name == "Question" {
				found = true
				return false
			}
		case *ast.Ident:
			if value.Name == "modifiedQuestion" {
				found = true
				return false
			}
		}
		return true
	})
	return found
}
