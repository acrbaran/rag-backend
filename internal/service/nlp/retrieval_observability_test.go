package nlp

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"testing"
)

func TestRetrievalLogsDoNotUseQuestionContent(t *testing.T) {
	function, fileSet := parseRetrievalSource(t)

	ast.Inspect(function.Body, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok || !isCommonRetrievalLogCall(call) {
			return true
		}
		for _, argument := range call.Args {
			if usesRetrievalQuestionContent(argument) {
				t.Errorf("Retrieval log at %s uses raw question content", fileSet.Position(call.Pos()))
				break
			}
		}
		return true
	})
}

func TestRetrievalLogsQuestionLength(t *testing.T) {
	function, _ := parseRetrievalSource(t)
	found := false
	ast.Inspect(function.Body, func(node ast.Node) bool {
		literal, ok := node.(*ast.BasicLit)
		if ok && literal.Kind == token.STRING && literal.Value == `"question_length"` {
			found = true
		}
		return true
	})
	if !found {
		t.Error("Retrieval logs missing safe metadata key \"question_length\"")
	}
}

func parseRetrievalSource(t *testing.T) (*ast.FuncDecl, *token.FileSet) {
	t.Helper()
	sourcePath := os.Getenv("RAGFLOW_RETRIEVAL_SOURCE")
	if sourcePath == "" {
		sourcePath = "retrieval.go"
	}
	fileSet := token.NewFileSet()
	file, err := parser.ParseFile(fileSet, sourcePath, nil, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", sourcePath, err)
	}
	for _, declaration := range file.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if ok && function.Name.Name == "Retrieval" {
			return function, fileSet
		}
	}
	t.Fatalf("Retrieval not found in %s", sourcePath)
	return nil, nil
}

func isCommonRetrievalLogCall(call *ast.CallExpr) bool {
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

func usesRetrievalQuestionContent(expression ast.Expr) bool {
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
		}
		return true
	})
	return found
}
