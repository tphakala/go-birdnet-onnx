# Go BirdNET ONNX Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Go library for running BirdNET v2.4/v3.0 and Google Perch v2 bird species classification inference using ONNX Runtime, with location/date-based species filtering via RangeFilter.

**Architecture:** Single flat `birdnet` package. Functional options API. `DynamicAdvancedSession` from `onnxruntime_go` for thread-safe inference. TDD throughout — all unit tests run without ONNX model files.

**Tech Stack:** Go 1.26, `github.com/yalue/onnxruntime_go`, `github.com/stretchr/testify` (testing)

**Spec:** `docs/superpowers/specs/2026-03-22-go-birdnet-onnx-design.md`

---

## File Structure

```
birdnet/
    birdnet.go          # package doc, MustInitORT, DestroyORT
    types.go            # ModelType, ModelConfig, Prediction, Result, LocationScore
    errors.go           # error types and sentinel errors
    postprocess.go      # sigmoid, softmax, topK
    labels.go           # loadLabels (text, CSV, JSON parsing)
    detection.go        # detectModelType, ModelConfig construction
    classifier.go       # Classifier, ClassifierOption, NewClassifier, Predict, PredictBatch
    rangefilter.go      # RangeFilter, RangeFilterOption, NewRangeFilter, week calculation
    types_test.go       # ModelType method tests
    errors_test.go      # error formatting tests
    postprocess_test.go # sigmoid, softmax, topK tests
    labels_test.go      # label parsing tests
    detection_test.go   # model detection tests
    rangefilter_test.go # week calculation, validation, filter logic tests
    classifier_test.go  # integration tests (build tag gated)
    go.mod
    go.sum
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `birdnet/go.mod`
- Create: `birdnet/birdnet.go`

- [ ] **Step 1: Initialize Go module**

```bash
cd birdnet && go mod init github.com/tphakala/birdnet-onnx && go get github.com/yalue/onnxruntime_go && go get github.com/stretchr/testify
```

- [ ] **Step 2: Create package entry point with ORT helpers**

Create `birdnet/birdnet.go`:

```go
// Package birdnet provides inference for BirdNET and Google Perch bird species
// classification models using ONNX Runtime.
//
// The caller is responsible for initializing the ONNX Runtime environment
// before creating any Classifier or RangeFilter instances. Use MustInitORT
// for simple applications, or call ort.SetSharedLibraryPath and
// ort.InitializeEnvironment directly for full control.
package birdnet

import ort "github.com/yalue/onnxruntime_go"

// MustInitORT initializes the ONNX Runtime with the given shared library path.
// It panics on failure. Intended for simple applications.
// For production use, call ort.SetSharedLibraryPath and ort.InitializeEnvironment directly.
func MustInitORT(libraryPath string) {
	ort.SetSharedLibraryPath(libraryPath)
	if err := ort.InitializeEnvironment(); err != nil {
		panic("birdnet: failed to initialize ONNX Runtime: " + err.Error())
	}
}

// DestroyORT tears down the ONNX Runtime environment.
// Call this when completely done with all classifiers and range filters.
func DestroyORT() {
	ort.DestroyEnvironment()
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cd birdnet && go build ./...`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add birdnet/go.mod birdnet/go.sum birdnet/birdnet.go
git commit -m "feat: scaffold birdnet package with ORT init helpers"
```

---

### Task 2: Types — ModelType, ModelConfig, Prediction, Result

**Files:**
- Create: `birdnet/types.go`
- Create: `birdnet/types_test.go`

- [ ] **Step 1: Write failing tests for ModelType methods**

Create `birdnet/types_test.go`:

```go
package birdnet

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestModelType_String(t *testing.T) {
	tests := []struct {
		mt   ModelType
		want string
	}{
		{BirdNETv24, "BirdNET v2.4"},
		{BirdNETv30, "BirdNET v3.0"},
		{PerchV2, "Perch v2"},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, tt.mt.String())
	}
}

func TestModelType_SampleRate(t *testing.T) {
	assert.Equal(t, 48000, BirdNETv24.SampleRate())
	assert.Equal(t, 32000, BirdNETv30.SampleRate())
	assert.Equal(t, 32000, PerchV2.SampleRate())
}

func TestModelType_Duration(t *testing.T) {
	assert.Equal(t, 3.0, BirdNETv24.Duration())
	assert.Equal(t, 5.0, BirdNETv30.Duration())
	assert.Equal(t, 5.0, PerchV2.Duration())
}

func TestModelType_SampleCount(t *testing.T) {
	assert.Equal(t, 144000, BirdNETv24.SampleCount())
	assert.Equal(t, 160000, BirdNETv30.SampleCount())
	assert.Equal(t, 160000, PerchV2.SampleCount())
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd birdnet && go test -run TestModelType -v`
Expected: compilation failure — types not defined

- [ ] **Step 3: Implement types**

Create `birdnet/types.go`:

```go
package birdnet

// ModelType identifies which bird classification model is being used.
type ModelType int

const (
	BirdNETv24 ModelType = iota
	BirdNETv30
	PerchV2
)

func (m ModelType) String() string {
	switch m {
	case BirdNETv24:
		return "BirdNET v2.4"
	case BirdNETv30:
		return "BirdNET v3.0"
	case PerchV2:
		return "Perch v2"
	default:
		return "Unknown"
	}
}

func (m ModelType) SampleRate() int {
	switch m {
	case BirdNETv24:
		return 48000
	case BirdNETv30, PerchV2:
		return 32000
	default:
		return 0
	}
}

func (m ModelType) Duration() float64 {
	switch m {
	case BirdNETv24:
		return 3.0
	case BirdNETv30, PerchV2:
		return 5.0
	default:
		return 0
	}
}

func (m ModelType) SampleCount() int {
	switch m {
	case BirdNETv24:
		return 144000
	case BirdNETv30, PerchV2:
		return 160000
	default:
		return 0
	}
}

// ModelConfig holds the derived parameters for a loaded model.
// Constructed internally during model loading.
type ModelConfig struct {
	Type          ModelType
	SampleRate    int
	Duration      float64
	SampleCount   int
	NumOutputs    int
	EmbeddingSize int     // 0 if model doesn't produce embeddings
	LogitsIndex   int     // which output tensor contains logits
	InputShape    []int64 // actual shape from ONNX model
}

// Prediction represents a single species prediction with confidence score.
type Prediction struct {
	Species    string
	Confidence float32
	Index      int
}

// Result holds the full output of a classification inference.
type Result struct {
	ModelType   ModelType
	Predictions []Prediction // top-K, sorted descending by confidence
	Embeddings  []float32    // nil if model doesn't produce embeddings
	RawScores   []float32    // all scores after activation (sigmoid/softmax)
}

// LocationScore represents a species' occurrence probability at a given location and date.
type LocationScore struct {
	Species string
	Score   float32
	Index   int
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd birdnet && go test -run TestModelType -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add birdnet/types.go birdnet/types_test.go
git commit -m "feat: add ModelType, ModelConfig, Prediction, Result types"
```

---

### Task 3: Error Types

**Files:**
- Create: `birdnet/errors.go`
- Create: `birdnet/errors_test.go`

- [ ] **Step 1: Write failing tests for error types**

Create `birdnet/errors_test.go`:

```go
package birdnet

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInputSizeError(t *testing.T) {
	err := &InputSizeError{Expected: 144000, Got: 100000}
	assert.Contains(t, err.Error(), "144000")
	assert.Contains(t, err.Error(), "100000")

	target, ok := errors.AsType[*InputSizeError](err)
	require.True(t, ok)
	assert.Equal(t, 144000, target.Expected)
}

func TestBatchInputSizeError(t *testing.T) {
	err := &BatchInputSizeError{Index: 2, Expected: 160000, Got: 50000}
	assert.Contains(t, err.Error(), "segment 2")
	assert.Contains(t, err.Error(), "160000")
	assert.Contains(t, err.Error(), "50000")
}

func TestLabelCountError(t *testing.T) {
	err := &LabelCountError{Expected: 6522, Got: 100}
	assert.Contains(t, err.Error(), "6522")
	assert.Contains(t, err.Error(), "100")
}

func TestModelDetectionError(t *testing.T) {
	err := &ModelDetectionError{Reason: "unsupported shape"}
	assert.Contains(t, err.Error(), "unsupported shape")
}

func TestLabelLoadError(t *testing.T) {
	err := &LabelLoadError{Path: "/tmp/labels.txt", Reason: "file not found"}
	assert.Contains(t, err.Error(), "/tmp/labels.txt")
	assert.Contains(t, err.Error(), "file not found")
}

func TestInvalidCoordinatesError(t *testing.T) {
	err := &InvalidCoordinatesError{Latitude: 91.0, Longitude: 0, Reason: "latitude out of range"}
	assert.Contains(t, err.Error(), "91")
	assert.Contains(t, err.Error(), "latitude out of range")
}

func TestInvalidDateError(t *testing.T) {
	err := &InvalidDateError{Month: 13, Day: 1, Reason: "month out of range"}
	assert.Contains(t, err.Error(), "13")
	assert.Contains(t, err.Error(), "month out of range")
}

func TestSentinelErrors(t *testing.T) {
	assert.ErrorIs(t, ErrModelPathRequired, ErrModelPathRequired)
	assert.ErrorIs(t, ErrLabelsRequired, ErrLabelsRequired)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd birdnet && go test -run "Test.*Error" -v`
Expected: compilation failure

- [ ] **Step 3: Implement error types**

Create `birdnet/errors.go`:

```go
package birdnet

import (
	"errors"
	"fmt"
)

var (
	ErrModelPathRequired = errors.New("birdnet: model path is required")
	ErrLabelsRequired    = errors.New("birdnet: labels are required")
	ErrEmptyBatch        = errors.New("birdnet: batch must contain at least one segment")
)

type InputSizeError struct {
	Expected int
	Got      int
}

func (e *InputSizeError) Error() string {
	return fmt.Sprintf("birdnet: expected %d audio samples, got %d", e.Expected, e.Got)
}

type BatchInputSizeError struct {
	Index    int
	Expected int
	Got      int
}

func (e *BatchInputSizeError) Error() string {
	return fmt.Sprintf("birdnet: segment %d has %d audio samples, expected %d", e.Index, e.Got, e.Expected)
}

type LabelCountError struct {
	Expected int
	Got      int
}

func (e *LabelCountError) Error() string {
	return fmt.Sprintf("birdnet: label count mismatch: model has %d classes but %d labels were provided", e.Expected, e.Got)
}

type ModelDetectionError struct {
	Reason string
}

func (e *ModelDetectionError) Error() string {
	return fmt.Sprintf("birdnet: cannot detect model type: %s", e.Reason)
}

type LabelLoadError struct {
	Path   string
	Reason string
}

func (e *LabelLoadError) Error() string {
	return fmt.Sprintf("birdnet: failed to load labels from %s: %s", e.Path, e.Reason)
}

type InvalidCoordinatesError struct {
	Latitude  float32
	Longitude float32
	Reason    string
}

func (e *InvalidCoordinatesError) Error() string {
	return fmt.Sprintf("birdnet: invalid coordinates (%.2f, %.2f): %s", e.Latitude, e.Longitude, e.Reason)
}

type InvalidDateError struct {
	Month  int
	Day    int
	Reason string
}

func (e *InvalidDateError) Error() string {
	return fmt.Sprintf("birdnet: invalid date (month=%d, day=%d): %s", e.Month, e.Day, e.Reason)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd birdnet && go test -run "Test.*Error|TestSentinel" -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add birdnet/errors.go birdnet/errors_test.go
git commit -m "feat: add error types and sentinel errors"
```

---

### Task 4: Post-Processing — sigmoid, softmax, topK

**Files:**
- Create: `birdnet/postprocess.go`
- Create: `birdnet/postprocess_test.go`

- [ ] **Step 1: Write failing tests for sigmoid**

Create `birdnet/postprocess_test.go`:

```go
package birdnet

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSigmoid(t *testing.T) {
	tests := []struct {
		input    float32
		expected float32
	}{
		{0, 0.5},
		{10, 0.99995},   // strong detection
		{-10, 0.0000454}, // non-detection
		{5, 0.9933},
		{-5, 0.00669},
	}
	for _, tt := range tests {
		got := sigmoid(tt.input)
		assert.InDelta(t, tt.expected, got, 0.001, "sigmoid(%f)", tt.input)
	}
}

func TestSigmoidSlice(t *testing.T) {
	input := []float32{0, 10, -10}
	result := sigmoidSlice(input)
	require.Len(t, result, 3)
	assert.InDelta(t, 0.5, result[0], 0.001)
	assert.InDelta(t, 1.0, result[1], 0.001)
	assert.InDelta(t, 0.0, result[2], 0.001)
	// original must not be modified
	assert.Equal(t, float32(0), input[0])
}

func TestSoftmax(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0}
	result := softmax(input)
	require.Len(t, result, 3)

	// probabilities must sum to 1
	var sum float32
	for _, v := range result {
		sum += v
	}
	assert.InDelta(t, 1.0, sum, 0.0001)

	// ordering preserved: result[2] > result[1] > result[0]
	assert.Greater(t, result[2], result[1])
	assert.Greater(t, result[1], result[0])
}

func TestSoftmax_NumericalStability(t *testing.T) {
	// large values should not cause overflow
	input := []float32{1000, 1001, 1002}
	result := softmax(input)
	for _, v := range result {
		assert.False(t, math.IsNaN(float64(v)), "softmax produced NaN")
		assert.False(t, math.IsInf(float64(v), 0), "softmax produced Inf")
	}
	var sum float32
	for _, v := range result {
		sum += v
	}
	assert.InDelta(t, 1.0, sum, 0.0001)
}

func TestSoftmax_SingleElement(t *testing.T) {
	result := softmax([]float32{42.0})
	require.Len(t, result, 1)
	assert.InDelta(t, 1.0, result[0], 0.0001)
}

func TestTopK(t *testing.T) {
	scores := []float32{0.1, 0.9, 0.5, 0.3, 0.7}
	labels := []string{"a", "b", "c", "d", "e"}
	result := topK(scores, labels, 3, 0.0)

	require.Len(t, result, 3)
	assert.Equal(t, "b", result[0].Species)
	assert.Equal(t, float32(0.9), result[0].Confidence)
	assert.Equal(t, 1, result[0].Index)
	assert.Equal(t, "e", result[1].Species)
	assert.Equal(t, "c", result[2].Species)
}

func TestTopK_MinConfidence(t *testing.T) {
	scores := []float32{0.1, 0.9, 0.5, 0.01, 0.7}
	labels := []string{"a", "b", "c", "d", "e"}
	result := topK(scores, labels, 10, 0.3)

	require.Len(t, result, 3)
	assert.Equal(t, "b", result[0].Species)
	assert.Equal(t, "e", result[1].Species)
	assert.Equal(t, "c", result[2].Species)
}

func TestTopK_KLargerThanInput(t *testing.T) {
	scores := []float32{0.5, 0.9}
	labels := []string{"a", "b"}
	result := topK(scores, labels, 10, 0.0)

	require.Len(t, result, 2)
	assert.Equal(t, "b", result[0].Species)
	assert.Equal(t, "a", result[1].Species)
}

func TestTopK_AllBelowMinConfidence(t *testing.T) {
	scores := []float32{0.01, 0.02}
	labels := []string{"a", "b"}
	result := topK(scores, labels, 10, 0.5)
	assert.Empty(t, result)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd birdnet && go test -run "TestSigmoid|TestSoftmax|TestTopK" -v`
Expected: compilation failure

- [ ] **Step 3: Implement post-processing functions**

Create `birdnet/postprocess.go`:

```go
package birdnet

import (
	"math"
	"slices"
)

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

func sigmoidSlice(logits []float32) []float32 {
	result := make([]float32, len(logits))
	for i, v := range logits {
		result[i] = sigmoid(v)
	}
	return result
}

func softmax(logits []float32) []float32 {
	result := make([]float32, len(logits))

	// find max for numerical stability
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	// exponentiate and sum
	var sum float32
	for i, v := range logits {
		result[i] = float32(math.Exp(float64(v - maxVal)))
		sum += result[i]
	}

	// normalize
	for i := range result {
		result[i] /= sum
	}
	return result
}

func topK(scores []float32, labels []string, k int, minConf float32) []Prediction {
	// collect predictions above minimum confidence
	var preds []Prediction
	for i, score := range scores {
		if score >= minConf {
			preds = append(preds, Prediction{
				Species:    labels[i],
				Confidence: score,
				Index:      i,
			})
		}
	}

	// sort descending by confidence
	slices.SortFunc(preds, func(a, b Prediction) int {
		if a.Confidence > b.Confidence {
			return -1
		}
		if a.Confidence < b.Confidence {
			return 1
		}
		return 0
	})

	// truncate to k
	if len(preds) > k {
		preds = preds[:k]
	}
	return preds
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd birdnet && go test -run "TestSigmoid|TestSoftmax|TestTopK" -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add birdnet/postprocess.go birdnet/postprocess_test.go
git commit -m "feat: add sigmoid, softmax, and topK post-processing"
```

---

### Task 5: Label Loading

**Files:**
- Create: `birdnet/labels.go`
- Create: `birdnet/labels_test.go`
- Create: `birdnet/testdata/labels.txt`
- Create: `birdnet/testdata/labels_semicolon.csv`
- Create: `birdnet/testdata/labels_comma.csv`
- Create: `birdnet/testdata/labels_array.json`
- Create: `birdnet/testdata/labels_object.json`
- Create: `birdnet/testdata/labels_named.json`

- [ ] **Step 1: Create test fixture files**

Create `birdnet/testdata/labels.txt`:
```
Turdus merula_Common Blackbird
Parus major_Great Tit
Erithacus rubecula_European Robin
```

Create `birdnet/testdata/labels_semicolon.csv`:
```
idx;sci_name;com_name;class;order
0;Turdus merula;Common Blackbird;Aves;Passeriformes
1;Parus major;Great Tit;Aves;Passeriformes
2;Erithacus rubecula;European Robin;Aves;Passeriformes
```

Create `birdnet/testdata/labels_comma.csv`:
```
idx,sci_name,com_name,class,order
0,Turdus merula,Common Blackbird,Aves,Passeriformes
1,Parus major,Great Tit,Aves,Passeriformes
2,Erithacus rubecula,European Robin,Aves,Passeriformes
```

Create `birdnet/testdata/labels_array.json`:
```json
["Turdus merula", "Parus major", "Erithacus rubecula"]
```

Create `birdnet/testdata/labels_object.json`:
```json
{"labels": ["Turdus merula", "Parus major", "Erithacus rubecula"]}
```

Create `birdnet/testdata/labels_named.json`:
```json
[{"name": "Turdus merula"}, {"name": "Parus major"}, {"name": "Erithacus rubecula"}]
```

- [ ] **Step 2: Write failing tests for label loading**

Create `birdnet/labels_test.go`:

```go
package birdnet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadLabels_Text(t *testing.T) {
	labels, err := loadLabels("testdata/labels.txt")
	require.NoError(t, err)
	require.Len(t, labels, 3)
	assert.Equal(t, "Turdus merula_Common Blackbird", labels[0])
	assert.Equal(t, "Parus major_Great Tit", labels[1])
	assert.Equal(t, "Erithacus rubecula_European Robin", labels[2])
}

func TestLoadLabels_Text_EmptyLines(t *testing.T) {
	// write temp file with empty lines
	content := "\n  Turdus merula  \n\n  Parus major  \n\n"
	labels, err := loadLabelsFromBytes([]byte(content), ".txt")
	require.NoError(t, err)
	require.Len(t, labels, 2)
	assert.Equal(t, "Turdus merula", labels[0])
	assert.Equal(t, "Parus major", labels[1])
}

func TestLoadLabels_CSV_Semicolon(t *testing.T) {
	labels, err := loadLabels("testdata/labels_semicolon.csv")
	require.NoError(t, err)
	require.Len(t, labels, 3)
	assert.Equal(t, "Turdus merula", labels[0])
	assert.Equal(t, "Parus major", labels[1])
}

func TestLoadLabels_CSV_Comma(t *testing.T) {
	labels, err := loadLabels("testdata/labels_comma.csv")
	require.NoError(t, err)
	require.Len(t, labels, 3)
	assert.Equal(t, "Turdus merula", labels[0])
}

func TestLoadLabels_JSON_Array(t *testing.T) {
	labels, err := loadLabels("testdata/labels_array.json")
	require.NoError(t, err)
	require.Len(t, labels, 3)
	assert.Equal(t, "Turdus merula", labels[0])
}

func TestLoadLabels_JSON_Object(t *testing.T) {
	labels, err := loadLabels("testdata/labels_object.json")
	require.NoError(t, err)
	require.Len(t, labels, 3)
	assert.Equal(t, "Turdus merula", labels[0])
}

func TestLoadLabels_JSON_Named(t *testing.T) {
	labels, err := loadLabels("testdata/labels_named.json")
	require.NoError(t, err)
	require.Len(t, labels, 3)
	assert.Equal(t, "Turdus merula", labels[0])
}

func TestLoadLabels_FileNotFound(t *testing.T) {
	_, err := loadLabels("testdata/nonexistent.txt")
	require.Error(t, err)
	var labelErr *LabelLoadError
	require.ErrorAs(t, err, &labelErr)
	assert.Contains(t, labelErr.Path, "nonexistent.txt")
}

func TestLoadLabels_UnsupportedExtension(t *testing.T) {
	_, err := loadLabels("testdata/labels.xyz")
	require.Error(t, err)
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd birdnet && go test -run TestLoadLabels -v`
Expected: compilation failure

- [ ] **Step 4: Implement label loading**

Create `birdnet/labels.go`:

```go
package birdnet

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func loadLabels(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, &LabelLoadError{Path: path, Reason: err.Error()}
	}
	ext := strings.ToLower(filepath.Ext(path))
	labels, err := loadLabelsFromBytes(data, ext)
	if err != nil {
		return nil, &LabelLoadError{Path: path, Reason: err.Error()}
	}
	return labels, nil
}

func loadLabelsFromBytes(data []byte, ext string) ([]string, error) {
	switch ext {
	case ".txt":
		return loadLabelsText(data)
	case ".csv":
		return loadLabelsCSV(data)
	case ".json":
		return loadLabelsJSON(data)
	default:
		return nil, &LabelLoadError{Path: "(bytes)", Reason: "unsupported label file extension: " + ext}
	}
}

func loadLabelsText(data []byte) ([]string, error) {
	var labels []string
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			labels = append(labels, line)
		}
	}
	return labels, scanner.Err()
}

func loadLabelsCSV(data []byte) ([]string, error) {
	// auto-detect delimiter by checking first line
	firstLine, _, _ := bytes.Cut(data, []byte("\n"))
	delimiter := ','
	if bytes.Count(firstLine, []byte(";")) > bytes.Count(firstLine, []byte(",")) {
		delimiter = ';'
	}

	r := csv.NewReader(bytes.NewReader(data))
	r.Comma = rune(delimiter)
	r.LazyQuotes = true

	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) < 2 {
		return nil, nil
	}

	// find the best column by header
	header := records[0]
	colIdx := findLabelColumn(header)

	var labels []string
	for _, row := range records[1:] {
		if colIdx < len(row) {
			label := strings.TrimSpace(row[colIdx])
			if label != "" {
				labels = append(labels, label)
			}
		}
	}
	return labels, nil
}

func findLabelColumn(header []string) int {
	// priority order for column names
	priorities := []string{"sci_name", "com_name", "name", "label"}
	for _, name := range priorities {
		for i, h := range header {
			if strings.EqualFold(strings.TrimSpace(h), name) {
				return i
			}
		}
	}
	// fall back to first non-numeric column
	for i, h := range header {
		h = strings.TrimSpace(h)
		if _, err := strconv.Atoi(h); err != nil {
			return i
		}
	}
	return 0
}

func loadLabelsJSON(data []byte) ([]string, error) {
	// try flat string array
	var arr []string
	if err := json.Unmarshal(data, &arr); err == nil {
		return arr, nil
	}

	// try object with "labels" key
	var obj struct {
		Labels []string `json:"labels"`
	}
	if err := json.Unmarshal(data, &obj); err == nil && len(obj.Labels) > 0 {
		return obj.Labels, nil
	}

	// try array of objects with "name" key
	var named []struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(data, &named); err == nil && len(named) > 0 {
		labels := make([]string, len(named))
		for i, n := range named {
			labels[i] = n.Name
		}
		return labels, nil
	}

	return nil, &LabelLoadError{Path: "(json)", Reason: "unrecognized JSON label format"}
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd birdnet && go test -run TestLoadLabels -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add birdnet/labels.go birdnet/labels_test.go birdnet/testdata/
git commit -m "feat: add label loading for text, CSV, and JSON formats"
```

---

### Task 6: Model Auto-Detection

**Files:**
- Create: `birdnet/detection.go`
- Create: `birdnet/detection_test.go`

- [ ] **Step 1: Write failing tests for model detection**

Create `birdnet/detection_test.go`:

```go
package birdnet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDetectModelType_BirdNETv24(t *testing.T) {
	mt, err := detectModelTypeFromShapes([][]int64{{1, 144000}}, 1)
	require.NoError(t, err)
	assert.Equal(t, BirdNETv24, mt)
}

func TestDetectModelType_BirdNETv24_3D(t *testing.T) {
	mt, err := detectModelTypeFromShapes([][]int64{{1, 1, 144000}}, 1)
	require.NoError(t, err)
	assert.Equal(t, BirdNETv24, mt)
}

func TestDetectModelType_BirdNETv30(t *testing.T) {
	mt, err := detectModelTypeFromShapes([][]int64{{1, 160000}}, 2)
	require.NoError(t, err)
	assert.Equal(t, BirdNETv30, mt)
}

func TestDetectModelType_PerchV2(t *testing.T) {
	mt, err := detectModelTypeFromShapes([][]int64{{1, 160000}}, 4)
	require.NoError(t, err)
	assert.Equal(t, PerchV2, mt)
}

func TestDetectModelType_UnknownSamples(t *testing.T) {
	_, err := detectModelTypeFromShapes([][]int64{{1, 100000}}, 1)
	require.Error(t, err)
	var detErr *ModelDetectionError
	require.ErrorAs(t, err, &detErr)
}

func TestDetectModelType_UnknownOutputCount(t *testing.T) {
	_, err := detectModelTypeFromShapes([][]int64{{1, 160000}}, 3)
	require.Error(t, err)
}

func TestDetectModelType_NoInputs(t *testing.T) {
	_, err := detectModelTypeFromShapes(nil, 1)
	require.Error(t, err)
}

func TestBuildModelConfig_BirdNETv24(t *testing.T) {
	cfg := buildModelConfig(BirdNETv24, []int64{1, 144000}, 1, 6522)
	assert.Equal(t, BirdNETv24, cfg.Type)
	assert.Equal(t, 48000, cfg.SampleRate)
	assert.Equal(t, 3.0, cfg.Duration)
	assert.Equal(t, 144000, cfg.SampleCount)
	assert.Equal(t, 0, cfg.EmbeddingSize)
	assert.Equal(t, 0, cfg.LogitsIndex)
}

func TestBuildModelConfig_BirdNETv30(t *testing.T) {
	cfg := buildModelConfig(BirdNETv30, []int64{1, 160000}, 2, 11560)
	assert.Equal(t, 1280, cfg.EmbeddingSize)
	assert.Equal(t, 1, cfg.LogitsIndex)
}

func TestBuildModelConfig_PerchV2(t *testing.T) {
	cfg := buildModelConfig(PerchV2, []int64{1, 160000}, 4, 14795)
	assert.Equal(t, 1536, cfg.EmbeddingSize)
	assert.Equal(t, 3, cfg.LogitsIndex)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd birdnet && go test -run "TestDetect|TestBuild" -v`
Expected: compilation failure

- [ ] **Step 3: Implement model detection**

Create `birdnet/detection.go`:

```go
package birdnet

import "fmt"

// detectModelTypeFromShapes determines the model type from input shapes and output count.
// inputShapes contains the shapes of all input tensors.
// numOutputs is the number of output tensors.
func detectModelTypeFromShapes(inputShapes [][]int64, numOutputs int) (ModelType, error) {
	if len(inputShapes) == 0 {
		return 0, &ModelDetectionError{Reason: "model has no input tensors"}
	}

	shape := inputShapes[0]
	if len(shape) < 2 {
		return 0, &ModelDetectionError{Reason: fmt.Sprintf("input shape has %d dimensions, expected at least 2", len(shape))}
	}

	// sample count is always the last dimension
	sampleCount := shape[len(shape)-1]

	switch {
	case sampleCount == 144000 && numOutputs == 1:
		return BirdNETv24, nil
	case sampleCount == 160000 && numOutputs == 2:
		return BirdNETv30, nil
	case sampleCount == 160000 && numOutputs == 4:
		return PerchV2, nil
	default:
		return 0, &ModelDetectionError{
			Reason: fmt.Sprintf("unrecognized model: %d input samples, %d outputs", sampleCount, numOutputs),
		}
	}
}

// buildModelConfig creates a ModelConfig from a detected model type and ONNX metadata.
func buildModelConfig(mt ModelType, inputShape []int64, numOutputs int, logitsSize int) ModelConfig {
	cfg := ModelConfig{
		Type:       mt,
		SampleRate: mt.SampleRate(),
		Duration:   mt.Duration(),
		SampleCount: mt.SampleCount(),
		NumOutputs: numOutputs,
		InputShape: make([]int64, len(inputShape)),
	}
	copy(cfg.InputShape, inputShape)

	switch mt {
	case BirdNETv24:
		cfg.LogitsIndex = 0
		cfg.EmbeddingSize = 0
	case BirdNETv30:
		cfg.LogitsIndex = 1
		cfg.EmbeddingSize = 1280
	case PerchV2:
		cfg.LogitsIndex = 3
		cfg.EmbeddingSize = 1536
	}

	return cfg
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd birdnet && go test -run "TestDetect|TestBuild" -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add birdnet/detection.go birdnet/detection_test.go
git commit -m "feat: add model type auto-detection from ONNX tensor shapes"
```

---

### Task 7: RangeFilter — Week Calculation, Validation, Filter Logic

**Files:**
- Create: `birdnet/rangefilter.go`
- Create: `birdnet/rangefilter_test.go`

This task implements all the pure logic in RangeFilter (no ONNX session). The ONNX session parts are added in Task 9.

- [ ] **Step 1: Write failing tests for week calculation and validation**

Create `birdnet/rangefilter_test.go`:

```go
package birdnet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCalculateWeek_January1(t *testing.T) {
	// (1-1)*4 + min((1-1)/7+1, 4) = 0 + 1 = 1
	assert.Equal(t, float32(1.0), CalculateWeek(1, 1))
}

func TestCalculateWeek_January8(t *testing.T) {
	// (1-1)*4 + min((8-1)/7+1, 4) = 0 + 2 = 2
	assert.Equal(t, float32(2.0), CalculateWeek(1, 8))
}

func TestCalculateWeek_January28(t *testing.T) {
	// (1-1)*4 + min((28-1)/7+1, 4) = 0 + 4 = 4
	assert.Equal(t, float32(4.0), CalculateWeek(1, 28))
}

func TestCalculateWeek_January31_Clamped(t *testing.T) {
	// (1-1)*4 + min((31-1)/7+1, 4) = 0 + min(5, 4) = 4
	assert.Equal(t, float32(4.0), CalculateWeek(1, 31))
}

func TestCalculateWeek_February1(t *testing.T) {
	// (2-1)*4 + min((1-1)/7+1, 4) = 4 + 1 = 5
	assert.Equal(t, float32(5.0), CalculateWeek(2, 1))
}

func TestCalculateWeek_December31_Clamped(t *testing.T) {
	// (12-1)*4 + min((31-1)/7+1, 4) = 44 + min(5, 4) = 48
	assert.Equal(t, float32(48.0), CalculateWeek(12, 31))
}

func TestCalculateWeek_December1(t *testing.T) {
	// (12-1)*4 + min((1-1)/7+1, 4) = 44 + 1 = 45
	assert.Equal(t, float32(45.0), CalculateWeek(12, 1))
}

func TestValidateCoordinates_Valid(t *testing.T) {
	assert.NoError(t, ValidateCoordinates(60.17, 24.94))
	assert.NoError(t, ValidateCoordinates(-90, -180))
	assert.NoError(t, ValidateCoordinates(90, 180))
	assert.NoError(t, ValidateCoordinates(0, 0))
}

func TestValidateCoordinates_Invalid(t *testing.T) {
	tests := []struct {
		lat, lon float32
	}{
		{91, 0},
		{-91, 0},
		{0, 181},
		{0, -181},
	}
	for _, tt := range tests {
		err := ValidateCoordinates(tt.lat, tt.lon)
		require.Error(t, err)
		var coordErr *InvalidCoordinatesError
		require.ErrorAs(t, err, &coordErr)
	}
}

func TestValidateDate_Valid(t *testing.T) {
	assert.NoError(t, ValidateDate(1, 1))
	assert.NoError(t, ValidateDate(12, 31))
	assert.NoError(t, ValidateDate(6, 15))
}

func TestValidateDate_Invalid(t *testing.T) {
	tests := []struct {
		month, day int
	}{
		{0, 1},
		{13, 1},
		{1, 0},
		{1, 32},
	}
	for _, tt := range tests {
		err := ValidateDate(tt.month, tt.day)
		require.Error(t, err)
		var dateErr *InvalidDateError
		require.ErrorAs(t, err, &dateErr)
	}
}

func TestFilter_Basic(t *testing.T) {
	predictions := []Prediction{
		{Species: "species_a", Confidence: 0.9, Index: 0},
		{Species: "species_b", Confidence: 0.8, Index: 1},
		{Species: "species_c", Confidence: 0.7, Index: 2},
	}
	scores := []LocationScore{
		{Species: "species_a", Score: 0.8, Index: 0},
		{Species: "species_b", Score: 0.01, Index: 1}, // below threshold
		{Species: "species_c", Score: 0.5, Index: 2},
	}

	result := filterPredictions(predictions, scores, 0.03, false)
	require.Len(t, result, 2)
	assert.Equal(t, "species_a", result[0].Species)
	assert.Equal(t, "species_c", result[1].Species)
}

func TestFilter_Rerank(t *testing.T) {
	predictions := []Prediction{
		{Species: "species_a", Confidence: 0.9, Index: 0},
		{Species: "species_b", Confidence: 0.8, Index: 1},
	}
	scores := []LocationScore{
		{Species: "species_a", Score: 0.1, Index: 0},
		{Species: "species_b", Score: 0.9, Index: 1},
	}

	result := filterPredictions(predictions, scores, 0.0, true)
	require.Len(t, result, 2)
	// species_b should now be first: 0.8*0.9 = 0.72 > 0.9*0.1 = 0.09
	assert.Equal(t, "species_b", result[0].Species)
	assert.InDelta(t, 0.72, result[0].Confidence, 0.001)
}

func TestFilterBatch(t *testing.T) {
	batches := [][]Prediction{
		{{Species: "a", Confidence: 0.9, Index: 0}},
		{{Species: "b", Confidence: 0.8, Index: 1}},
	}
	scores := []LocationScore{
		{Species: "a", Score: 0.5, Index: 0},
		{Species: "b", Score: 0.01, Index: 1},
	}

	result := filterBatchPredictions(batches, scores, 0.03, false)
	require.Len(t, result, 2)
	require.Len(t, result[0], 1) // species_a passes
	require.Len(t, result[1], 0) // species_b filtered out
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd birdnet && go test -run "TestCalculateWeek|TestValidate|TestFilter" -v`
Expected: compilation failure

- [ ] **Step 3: Implement RangeFilter pure logic**

Create `birdnet/rangefilter.go`:

```go
package birdnet

import (
	"slices"
)

// CalculateWeek returns the BirdNET 48-week year week number for a given month and day.
// BirdNET assumes 4 weeks per month. Days 29-31 are clamped to week 4 of their month.
// Result is always in [1, 48].
func CalculateWeek(month, day int) float32 {
	weeksFromMonths := (month - 1) * 4
	weekInMonth := (day - 1) / 7 + 1
	if weekInMonth > 4 {
		weekInMonth = 4
	}
	return float32(weeksFromMonths + weekInMonth)
}

// ValidateCoordinates checks that latitude is in [-90, 90] and longitude is in [-180, 180].
func ValidateCoordinates(latitude, longitude float32) error {
	if latitude < -90 || latitude > 90 {
		return &InvalidCoordinatesError{
			Latitude: latitude, Longitude: longitude,
			Reason: "latitude must be between -90 and 90",
		}
	}
	if longitude < -180 || longitude > 180 {
		return &InvalidCoordinatesError{
			Latitude: latitude, Longitude: longitude,
			Reason: "longitude must be between -180 and 180",
		}
	}
	return nil
}

// ValidateDate checks that month is in [1, 12] and day is in [1, 31].
func ValidateDate(month, day int) error {
	if month < 1 || month > 12 {
		return &InvalidDateError{Month: month, Day: day, Reason: "month must be between 1 and 12"}
	}
	if day < 1 || day > 31 {
		return &InvalidDateError{Month: month, Day: day, Reason: "day must be between 1 and 31"}
	}
	return nil
}

// filterPredictions removes predictions for species with location scores below threshold.
// If rerank is true, confidence is multiplied by location score and results are re-sorted.
func filterPredictions(predictions []Prediction, scores []LocationScore, threshold float32, rerank bool) []Prediction {
	// build score lookup by species
	scoreMap := make(map[string]float32, len(scores))
	for _, s := range scores {
		scoreMap[s.Species] = s.Score
	}

	var result []Prediction
	for _, p := range predictions {
		locScore, ok := scoreMap[p.Species]
		if !ok || locScore < threshold {
			continue
		}
		pred := p
		if rerank {
			pred.Confidence = p.Confidence * locScore
		}
		result = append(result, pred)
	}

	if rerank {
		slices.SortFunc(result, func(a, b Prediction) int {
			if a.Confidence > b.Confidence {
				return -1
			}
			if a.Confidence < b.Confidence {
				return 1
			}
			return 0
		})
	}
	return result
}

// filterBatchPredictions applies filterPredictions to each batch of predictions.
func filterBatchPredictions(batches [][]Prediction, scores []LocationScore, threshold float32, rerank bool) [][]Prediction {
	result := make([][]Prediction, len(batches))
	for i, batch := range batches {
		result[i] = filterPredictions(batch, scores, threshold, rerank)
	}
	return result
}

// RangeFilter uses the BirdNET meta model to filter species by geographic location and date.
// The ONNX session fields and methods (NewRangeFilter, Predict, Close) are added in Task 9.
type RangeFilter struct {
	session   interface{} // placeholder, replaced in Task 9 with *ort.DynamicAdvancedSession
	labels    []string
	threshold float32
	inputName  string
	outputName string
}

// RangeFilterOption configures a RangeFilter.
type RangeFilterOption func(*rangeFilterConfig)

type rangeFilterConfig struct {
	labels    []string
	labelsPath string
	threshold float32
}

// WithRangeFilterLabels provides labels directly.
func WithRangeFilterLabels(labels []string) RangeFilterOption {
	return func(c *rangeFilterConfig) { c.labels = labels }
}

// WithRangeFilterLabelsPath loads labels from a file.
func WithRangeFilterLabelsPath(path string) RangeFilterOption {
	return func(c *rangeFilterConfig) { c.labelsPath = path }
}

// WithRangeFilterThreshold sets the minimum location score for filtering.
func WithRangeFilterThreshold(threshold float32) RangeFilterOption {
	return func(c *rangeFilterConfig) { c.threshold = threshold }
}

// WithRangeFilterFromClassifierLabels uses the same labels as a Classifier.
func WithRangeFilterFromClassifierLabels(labels []string) RangeFilterOption {
	return func(c *rangeFilterConfig) {
		cp := make([]string, len(labels))
		copy(cp, labels)
		c.labels = cp
	}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd birdnet && go test -run "TestCalculateWeek|TestValidate|TestFilter" -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add birdnet/rangefilter.go birdnet/rangefilter_test.go
git commit -m "feat: add week calculation, validation, and filter logic"
```

---

### Task 8: Classifier — Construction, Predict, PredictBatch

**Files:**
- Create: `birdnet/classifier.go`

This is the core ONNX integration task. It creates the Classifier struct with NewClassifier, Predict, PredictBatch, and Close.

- [ ] **Step 1: Implement the Classifier**

Create `birdnet/classifier.go`:

```go
package birdnet

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// Classifier runs bird species classification inference on audio segments.
// It is safe for concurrent use from multiple goroutines.
type Classifier struct {
	session     *ort.DynamicAdvancedSession
	config      ModelConfig
	labels      []string
	topK        int
	minConf     float32
	inputName   string
	outputNames []string
}

// ClassifierOption configures classifier behavior.
type ClassifierOption func(*classifierConfig)

type classifierConfig struct {
	modelType      *ModelType
	labels         []string
	labelsPath     string
	topK           int
	minConf        float32
	sessionOptsFn  func(*ort.SessionOptions)
}

func defaultClassifierConfig() *classifierConfig {
	return &classifierConfig{
		topK:    10,
		minConf: 0.0,
	}
}

// WithModelType overrides auto-detection of the model type.
func WithModelType(t ModelType) ClassifierOption {
	return func(c *classifierConfig) { c.modelType = &t }
}

// WithLabels provides species labels directly.
func WithLabels(labels []string) ClassifierOption {
	return func(c *classifierConfig) { c.labels = labels }
}

// WithLabelsPath loads species labels from a file (text, CSV, or JSON).
func WithLabelsPath(path string) ClassifierOption {
	return func(c *classifierConfig) { c.labelsPath = path }
}

// WithTopK sets how many top predictions to return. Default: 10.
func WithTopK(k int) ClassifierOption {
	return func(c *classifierConfig) { c.topK = k }
}

// WithMinConfidence sets the minimum confidence threshold. Default: 0.0.
func WithMinConfidence(threshold float32) ClassifierOption {
	return func(c *classifierConfig) { c.minConf = threshold }
}

// WithSessionOptions provides a callback to configure the ONNX Runtime session options.
// The callback receives the options after defaults (IntraOpNumThreads=1, InterOpNumThreads=1)
// have been set, allowing the caller to override or add execution providers.
func WithSessionOptions(fn func(*ort.SessionOptions)) ClassifierOption {
	return func(c *classifierConfig) { c.sessionOptsFn = fn }
}

// NewClassifier creates a new Classifier from an ONNX model file.
// Model type is auto-detected from tensor shapes unless overridden with WithModelType.
// Labels must be provided via WithLabels or WithLabelsPath.
func NewClassifier(modelPath string, opts ...ClassifierOption) (*Classifier, error) {
	if modelPath == "" {
		return nil, ErrModelPathRequired
	}

	cfg := defaultClassifierConfig()
	for _, opt := range opts {
		opt(cfg)
	}

	// 1. Load model metadata to get input/output names and shapes
	info, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, fmt.Errorf("birdnet: failed to load model metadata: %w", err)
	}

	// 2. Extract input/output names and shapes
	inputNames := info.Inputs.GetNames()
	outputNames := info.Outputs.GetNames()
	if len(inputNames) == 0 {
		return nil, &ModelDetectionError{Reason: "model has no input tensors"}
	}

	inputShapes := make([][]int64, len(inputNames))
	for i := range inputNames {
		inputShapes[i] = info.Inputs[i].Dimensions
	}

	// 3. Validate input shape has enough dimensions
	if len(inputShapes[0]) < 2 {
		return nil, &ModelDetectionError{Reason: fmt.Sprintf("input shape has %d dimensions, expected at least 2", len(inputShapes[0]))}
	}

	// 4. Detect or use provided model type
	var mt ModelType
	if cfg.modelType != nil {
		mt = *cfg.modelType
	} else {
		mt, err = detectModelTypeFromShapes(inputShapes, len(outputNames))
		if err != nil {
			return nil, err
		}
	}

	// 5. Build model config
	modelCfg := buildModelConfig(mt, inputShapes[0], len(outputNames), 0)

	// 6. Load labels
	labels, err := resolveLabels(cfg)
	if err != nil {
		return nil, err
	}

	// 7. Validate label count against logits output size
	// Note: we get logits size from the output shape if available
	outputShapes := info.Outputs
	if modelCfg.LogitsIndex < len(outputShapes) {
		logitsDims := outputShapes[modelCfg.LogitsIndex].Dimensions
		if len(logitsDims) >= 2 {
			logitsSize := int(logitsDims[len(logitsDims)-1])
			if logitsSize > 0 && len(labels) != logitsSize {
				return nil, &LabelCountError{Expected: logitsSize, Got: len(labels)}
			}
		}
	}

	// 8. Create session options with defaults
	sessOpts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("birdnet: failed to create session options: %w", err)
	}
	if err := sessOpts.SetIntraOpNumThreads(1); err != nil {
		sessOpts.Destroy()
		return nil, fmt.Errorf("birdnet: failed to set intra-op threads: %w", err)
	}
	if err := sessOpts.SetInterOpNumThreads(1); err != nil {
		sessOpts.Destroy()
		return nil, fmt.Errorf("birdnet: failed to set inter-op threads: %w", err)
	}

	// 9. Apply user session options callback
	if cfg.sessionOptsFn != nil {
		cfg.sessionOptsFn(sessOpts)
	}

	// 10. Create DynamicAdvancedSession
	session, err := ort.NewDynamicAdvancedSession(modelPath,
		inputNames, outputNames, sessOpts)

	// 11. Destroy session options immediately (session doesn't take ownership)
	sessOpts.Destroy()

	if err != nil {
		return nil, fmt.Errorf("birdnet: failed to create ONNX session: %w", err)
	}

	return &Classifier{
		session:     session,
		config:      modelCfg,
		labels:      labels,
		topK:        cfg.topK,
		minConf:     cfg.minConf,
		inputName:   inputNames[0],
		outputNames: outputNames,
	}, nil
}

func resolveLabels(cfg *classifierConfig) ([]string, error) {
	if len(cfg.labels) > 0 {
		return cfg.labels, nil
	}
	if cfg.labelsPath != "" {
		return loadLabels(cfg.labelsPath)
	}
	return nil, ErrLabelsRequired
}

// Config returns the model configuration.
func (c *Classifier) Config() ModelConfig {
	return c.config
}

// Labels returns the species labels.
func (c *Classifier) Labels() []string {
	return c.labels
}

// Predict runs inference on a single audio segment.
// The audio slice must contain exactly Config().SampleCount float32 samples
// (mono, at the model's sample rate, normalized to [-1.0, 1.0]).
func (c *Classifier) Predict(audio []float32) (*Result, error) {
	if len(audio) != c.config.SampleCount {
		return nil, &InputSizeError{Expected: c.config.SampleCount, Got: len(audio)}
	}

	// Build input shape for batch size 1
	inputShape := makeBatchInputShape(c.config.InputShape, 1)

	// Create input tensor (wraps Go slice, no copy)
	inputTensor, err := ort.NewTensor(ort.NewShape(inputShape...), audio)
	if err != nil {
		return nil, fmt.Errorf("birdnet: failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensors
	outputs, err := c.createOutputTensors(1)
	if err != nil {
		return nil, err
	}
	defer destroyTensors(outputs)

	// Run inference
	err = c.session.Run(
		[]ort.Value{inputTensor},
		outputs,
	)
	if err != nil {
		return nil, fmt.Errorf("birdnet: inference failed: %w", err)
	}

	// Process results
	return c.processOutput(outputs, 0)
}

// PredictBatch runs inference on multiple audio segments in a single batch.
// Each segment must contain exactly Config().SampleCount samples.
func (c *Classifier) PredictBatch(segments [][]float32) ([]*Result, error) {
	if len(segments) == 0 {
		return nil, ErrEmptyBatch
	}

	// Validate all segment sizes
	for i, seg := range segments {
		if len(seg) != c.config.SampleCount {
			return nil, &BatchInputSizeError{Index: i, Expected: c.config.SampleCount, Got: len(seg)}
		}
	}

	batchSize := len(segments)

	// Concatenate into flat slice
	flat := make([]float32, 0, batchSize*c.config.SampleCount)
	for _, seg := range segments {
		flat = append(flat, seg...)
	}

	// Build input shape for batch
	inputShape := makeBatchInputShape(c.config.InputShape, int64(batchSize))

	// Create input tensor
	inputTensor, err := ort.NewTensor(ort.NewShape(inputShape...), flat)
	if err != nil {
		return nil, fmt.Errorf("birdnet: failed to create batch input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensors
	outputs, err := c.createOutputTensors(batchSize)
	if err != nil {
		return nil, err
	}
	defer destroyTensors(outputs)

	// Run inference
	err = c.session.Run(
		[]ort.Value{inputTensor},
		outputs,
	)
	if err != nil {
		return nil, fmt.Errorf("birdnet: batch inference failed: %w", err)
	}

	// Process results for each segment in the batch
	results := make([]*Result, batchSize)
	for i := range batchSize {
		results[i], err = c.processOutput(outputs, i)
		if err != nil {
			return nil, fmt.Errorf("birdnet: failed to process batch output %d: %w", i, err)
		}
	}
	return results, nil
}

// Close releases the ONNX session and associated resources.
func (c *Classifier) Close() error {
	if c.session != nil {
		c.session.Destroy()
		c.session = nil
	}
	return nil
}

// makeBatchInputShape replaces the batch dimension (first element) in the input shape.
func makeBatchInputShape(modelShape []int64, batchSize int64) []int64 {
	shape := make([]int64, len(modelShape))
	copy(shape, modelShape)
	shape[0] = batchSize
	return shape
}

// createOutputTensors creates pre-sized output tensors based on model config.
func (c *Classifier) createOutputTensors(batchSize int) ([]ort.Value, error) {
	outputs := make([]ort.Value, c.config.NumOutputs)
	for i := range c.config.NumOutputs {
		shape, err := c.outputShape(i, batchSize)
		if err != nil {
			destroyTensors(outputs)
			return nil, err
		}
		t, err := ort.NewEmptyTensor[float32](ort.NewShape(shape...))
		if err != nil {
			destroyTensors(outputs)
			return nil, fmt.Errorf("birdnet: failed to create output tensor %d: %w", i, err)
		}
		outputs[i] = t
	}
	return outputs, nil
}

// outputShape returns the expected shape for a given output tensor index.
func (c *Classifier) outputShape(outputIdx int, batchSize int) ([]int64, error) {
	batch := int64(batchSize)
	switch c.config.Type {
	case BirdNETv24:
		// 1 output: logits [batch, numClasses]
		return []int64{batch, int64(len(c.labels))}, nil
	case BirdNETv30:
		switch outputIdx {
		case 0: // embeddings [batch, 1280]
			return []int64{batch, int64(c.config.EmbeddingSize)}, nil
		case 1: // logits [batch, numClasses]
			return []int64{batch, int64(len(c.labels))}, nil
		}
	case PerchV2:
		switch outputIdx {
		case 0: // embedding [batch, 1536]
			return []int64{batch, int64(c.config.EmbeddingSize)}, nil
		case 1: // spatial embedding [batch, 16, 4, 1536]
			return []int64{batch, 16, 4, int64(c.config.EmbeddingSize)}, nil
		case 2: // spectrogram [batch, 500, 128]
			return []int64{batch, 500, 128}, nil
		case 3: // logits [batch, numClasses]
			return []int64{batch, int64(len(c.labels))}, nil
		}
	}
	return nil, fmt.Errorf("birdnet: unexpected output index %d for model %s", outputIdx, c.config.Type)
}

// processOutput extracts predictions and embeddings from output tensors for a given batch index.
func (c *Classifier) processOutput(outputs []ort.Value, batchIdx int) (*Result, error) {
	// Extract logits
	logitsTensor, ok := outputs[c.config.LogitsIndex].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("birdnet: logits tensor has unexpected type")
	}
	allLogits := logitsTensor.GetData()
	numClasses := len(c.labels)
	start := batchIdx * numClasses
	end := start + numClasses
	if end > len(allLogits) {
		return nil, fmt.Errorf("birdnet: logits tensor too small for batch index %d", batchIdx)
	}
	logits := allLogits[start:end]

	// Apply activation
	var scores []float32
	switch c.config.Type {
	case BirdNETv24, BirdNETv30:
		scores = sigmoidSlice(logits)
	case PerchV2:
		scores = softmax(logits)
	}

	// Extract embeddings if available
	var embeddings []float32
	if c.config.EmbeddingSize > 0 {
		embTensor, ok := outputs[0].(*ort.Tensor[float32])
		if ok {
			allEmb := embTensor.GetData()
			embStart := batchIdx * c.config.EmbeddingSize
			embEnd := embStart + c.config.EmbeddingSize
			if embEnd <= len(allEmb) {
				embeddings = make([]float32, c.config.EmbeddingSize)
				copy(embeddings, allEmb[embStart:embEnd])
			}
		}
	}

	// Top-K selection
	predictions := topK(scores, c.labels, c.topK, c.minConf)

	// scores is already a new Go slice (allocated by sigmoidSlice/softmax),
	// safe to use after tensor destroy
	return &Result{
		ModelType:   c.config.Type,
		Predictions: predictions,
		Embeddings:  embeddings,
		RawScores:   scores,
	}, nil
}

// destroyTensors destroys all non-nil tensors in the slice.
func destroyTensors(tensors []ort.Value) {
	for _, t := range tensors {
		if t != nil {
			t.Destroy()
		}
	}
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd birdnet && go build ./...`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add birdnet/classifier.go
git commit -m "feat: add Classifier with Predict and PredictBatch"
```

---

### Task 9: RangeFilter — ONNX Session Integration

**Files:**
- Modify: `birdnet/rangefilter.go`

Replace the placeholder session field with the real DynamicAdvancedSession and implement NewRangeFilter, Predict, Filter, FilterBatch, and Close.

- [ ] **Step 1: Update RangeFilter with ONNX session**

Replace the placeholder `RangeFilter` struct and add the ONNX methods in `birdnet/rangefilter.go`. Replace the `session interface{}` field and add `NewRangeFilter`, `Predict`, `Filter`, `FilterBatch`, `Close`:

```go
// Replace the RangeFilter struct with:
type RangeFilter struct {
	session    *ort.DynamicAdvancedSession
	labels     []string
	threshold  float32
	inputName  string
	outputName string
}

// Add import for ort at the top of the file:
// import ort "github.com/yalue/onnxruntime_go"

// NewRangeFilter creates a new RangeFilter from a BirdNET meta model ONNX file.
func NewRangeFilter(modelPath string, opts ...RangeFilterOption) (*RangeFilter, error) {
	if modelPath == "" {
		return nil, ErrModelPathRequired
	}

	cfg := &rangeFilterConfig{threshold: 0.03}
	for _, opt := range opts {
		opt(cfg)
	}

	// Load model metadata
	info, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, fmt.Errorf("birdnet: failed to load range filter model metadata: %w", err)
	}

	inputNames := info.Inputs.GetNames()
	outputNames := info.Outputs.GetNames()
	if len(inputNames) == 0 || len(outputNames) == 0 {
		return nil, &ModelDetectionError{Reason: "range filter model has no input or output tensors"}
	}

	// Resolve labels
	labels, err := resolveRangeFilterLabels(cfg)
	if err != nil {
		return nil, err
	}

	// Create session options
	sessOpts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("birdnet: failed to create session options: %w", err)
	}
	if err := sessOpts.SetIntraOpNumThreads(1); err != nil {
		sessOpts.Destroy()
		return nil, fmt.Errorf("birdnet: failed to set intra-op threads: %w", err)
	}
	if err := sessOpts.SetInterOpNumThreads(1); err != nil {
		sessOpts.Destroy()
		return nil, fmt.Errorf("birdnet: failed to set inter-op threads: %w", err)
	}

	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, sessOpts)
	sessOpts.Destroy()
	if err != nil {
		return nil, fmt.Errorf("birdnet: failed to create range filter session: %w", err)
	}

	return &RangeFilter{
		session:    session,
		labels:     labels,
		threshold:  cfg.threshold,
		inputName:  inputNames[0],
		outputName: outputNames[0],
	}, nil
}

func resolveRangeFilterLabels(cfg *rangeFilterConfig) ([]string, error) {
	if len(cfg.labels) > 0 {
		return cfg.labels, nil
	}
	if cfg.labelsPath != "" {
		return loadLabels(cfg.labelsPath)
	}
	return nil, ErrLabelsRequired
}

// Predict runs the meta model to get species occurrence probabilities for a location and date.
func (r *RangeFilter) Predict(latitude, longitude float32, month, day int) ([]LocationScore, error) {
	if err := ValidateCoordinates(latitude, longitude); err != nil {
		return nil, err
	}
	if err := ValidateDate(month, day); err != nil {
		return nil, err
	}

	week := CalculateWeek(month, day)
	input := []float32{latitude, longitude, week}

	inputTensor, err := ort.NewTensor(ort.NewShape(1, 3), input)
	if err != nil {
		return nil, fmt.Errorf("birdnet: failed to create range filter input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(len(r.labels))))
	if err != nil {
		return nil, fmt.Errorf("birdnet: failed to create range filter output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	err = r.session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	if err != nil {
		return nil, fmt.Errorf("birdnet: range filter inference failed: %w", err)
	}

	data := outputTensor.GetData()
	scores := make([]LocationScore, len(r.labels))
	for i, label := range r.labels {
		score := float32(0)
		if i < len(data) {
			score = data[i]
		}
		scores[i] = LocationScore{
			Species: label,
			Score:   score,
			Index:   i,
		}
	}
	return scores, nil
}

// Filter removes predictions for species below the location score threshold.
// If rerank is true, confidence is multiplied by location score and re-sorted.
func (r *RangeFilter) Filter(predictions []Prediction, scores []LocationScore, rerank bool) []Prediction {
	return filterPredictions(predictions, scores, r.threshold, rerank)
}

// FilterBatch applies Filter to each batch of predictions.
func (r *RangeFilter) FilterBatch(batches [][]Prediction, scores []LocationScore, rerank bool) [][]Prediction {
	return filterBatchPredictions(batches, scores, r.threshold, rerank)
}

// Close releases the ONNX session and associated resources.
func (r *RangeFilter) Close() error {
	if r.session != nil {
		r.session.Destroy()
		r.session = nil
	}
	return nil
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd birdnet && go build ./...`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add birdnet/rangefilter.go
git commit -m "feat: add RangeFilter with ONNX session integration"
```

---

### Task 10: Integration Tests

**Files:**
- Create: `birdnet/classifier_test.go`

Integration tests require ONNX model files. They are gated behind `//go:build integration`.

- [ ] **Step 1: Create integration test file**

Create `birdnet/classifier_test.go`:

```go
//go:build integration

package birdnet

import (
	"os"
	"runtime"
	"sync"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Set these environment variables to point to your model and label files:
//   BIRDNET_V24_MODEL, BIRDNET_V24_LABELS
//   BIRDNET_V30_MODEL, BIRDNET_V30_LABELS
//   PERCH_V2_MODEL, PERCH_V2_LABELS
//   BIRDNET_META_MODEL

func TestMain(m *testing.M) {
	libPath := os.Getenv("ORT_LIB_PATH")
	if libPath == "" {
		libPath = "onnxruntime.so"
	}
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		panic("failed to init ORT: " + err.Error())
	}
	code := m.Run()
	ort.DestroyEnvironment()
	os.Exit(code)
}

func TestClassifier_BirdNETv24(t *testing.T) {
	modelPath := os.Getenv("BIRDNET_V24_MODEL")
	labelsPath := os.Getenv("BIRDNET_V24_LABELS")
	if modelPath == "" || labelsPath == "" {
		t.Skip("BIRDNET_V24_MODEL or BIRDNET_V24_LABELS not set")
	}

	c, err := NewClassifier(modelPath, WithLabelsPath(labelsPath))
	require.NoError(t, err)
	defer c.Close()

	assert.Equal(t, BirdNETv24, c.Config().Type)
	assert.Equal(t, 48000, c.Config().SampleRate)
	assert.Equal(t, 144000, c.Config().SampleCount)

	// Predict with silence
	audio := make([]float32, 144000)
	result, err := c.Predict(audio)
	require.NoError(t, err)
	assert.NotNil(t, result)
	assert.Nil(t, result.Embeddings)
	assert.Len(t, result.RawScores, len(c.Labels()))
}

func TestClassifier_BirdNETv30(t *testing.T) {
	modelPath := os.Getenv("BIRDNET_V30_MODEL")
	labelsPath := os.Getenv("BIRDNET_V30_LABELS")
	if modelPath == "" || labelsPath == "" {
		t.Skip("BIRDNET_V30_MODEL or BIRDNET_V30_LABELS not set")
	}

	c, err := NewClassifier(modelPath, WithLabelsPath(labelsPath))
	require.NoError(t, err)
	defer c.Close()

	assert.Equal(t, BirdNETv30, c.Config().Type)
	assert.Equal(t, 1280, c.Config().EmbeddingSize)

	audio := make([]float32, 160000)
	result, err := c.Predict(audio)
	require.NoError(t, err)
	assert.NotNil(t, result.Embeddings)
	assert.Len(t, result.Embeddings, 1280)
}

func TestClassifier_PerchV2(t *testing.T) {
	modelPath := os.Getenv("PERCH_V2_MODEL")
	labelsPath := os.Getenv("PERCH_V2_LABELS")
	if modelPath == "" || labelsPath == "" {
		t.Skip("PERCH_V2_MODEL or PERCH_V2_LABELS not set")
	}

	c, err := NewClassifier(modelPath, WithLabelsPath(labelsPath))
	require.NoError(t, err)
	defer c.Close()

	assert.Equal(t, PerchV2, c.Config().Type)
	assert.Equal(t, 1536, c.Config().EmbeddingSize)

	audio := make([]float32, 160000)
	result, err := c.Predict(audio)
	require.NoError(t, err)
	assert.NotNil(t, result.Embeddings)
	assert.Len(t, result.Embeddings, 1536)
}

func TestClassifier_InputSizeError(t *testing.T) {
	modelPath := os.Getenv("BIRDNET_V24_MODEL")
	labelsPath := os.Getenv("BIRDNET_V24_LABELS")
	if modelPath == "" || labelsPath == "" {
		t.Skip("BIRDNET_V24_MODEL or BIRDNET_V24_LABELS not set")
	}

	c, err := NewClassifier(modelPath, WithLabelsPath(labelsPath))
	require.NoError(t, err)
	defer c.Close()

	audio := make([]float32, 100)
	_, err = c.Predict(audio)
	require.Error(t, err)
	var sizeErr *InputSizeError
	require.ErrorAs(t, err, &sizeErr)
	assert.Equal(t, 144000, sizeErr.Expected)
}

func TestClassifier_PredictBatch(t *testing.T) {
	modelPath := os.Getenv("BIRDNET_V24_MODEL")
	labelsPath := os.Getenv("BIRDNET_V24_LABELS")
	if modelPath == "" || labelsPath == "" {
		t.Skip("BIRDNET_V24_MODEL or BIRDNET_V24_LABELS not set")
	}

	c, err := NewClassifier(modelPath, WithLabelsPath(labelsPath))
	require.NoError(t, err)
	defer c.Close()

	seg1 := make([]float32, 144000)
	seg2 := make([]float32, 144000)
	results, err := c.PredictBatch([][]float32{seg1, seg2})
	require.NoError(t, err)
	require.Len(t, results, 2)
}

func TestClassifier_PredictBatch_EmptyBatch(t *testing.T) {
	modelPath := os.Getenv("BIRDNET_V24_MODEL")
	labelsPath := os.Getenv("BIRDNET_V24_LABELS")
	if modelPath == "" || labelsPath == "" {
		t.Skip("BIRDNET_V24_MODEL or BIRDNET_V24_LABELS not set")
	}

	c, err := NewClassifier(modelPath, WithLabelsPath(labelsPath))
	require.NoError(t, err)
	defer c.Close()

	_, err = c.PredictBatch(nil)
	require.ErrorIs(t, err, ErrEmptyBatch)
}

func TestClassifier_ConcurrentPredict(t *testing.T) {
	modelPath := os.Getenv("BIRDNET_V24_MODEL")
	labelsPath := os.Getenv("BIRDNET_V24_LABELS")
	if modelPath == "" || labelsPath == "" {
		t.Skip("BIRDNET_V24_MODEL or BIRDNET_V24_LABELS not set")
	}

	c, err := NewClassifier(modelPath, WithLabelsPath(labelsPath))
	require.NoError(t, err)
	defer c.Close()

	numGoroutines := runtime.NumCPU()
	var wg sync.WaitGroup
	errs := make([]error, numGoroutines)

	for i := range numGoroutines {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			audio := make([]float32, 144000)
			_, errs[idx] = c.Predict(audio)
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		assert.NoError(t, err, "goroutine %d failed", i)
	}
}

func TestClassifier_MemoryStability(t *testing.T) {
	modelPath := os.Getenv("BIRDNET_V24_MODEL")
	labelsPath := os.Getenv("BIRDNET_V24_LABELS")
	if modelPath == "" || labelsPath == "" {
		t.Skip("BIRDNET_V24_MODEL or BIRDNET_V24_LABELS not set")
	}

	c, err := NewClassifier(modelPath, WithLabelsPath(labelsPath))
	require.NoError(t, err)
	defer c.Close()

	audio := make([]float32, 144000)

	// Warm up
	for range 10 {
		_, err := c.Predict(audio)
		require.NoError(t, err)
	}
	runtime.GC()

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	baseAlloc := m.TotalAlloc

	// Run many iterations
	for range 100 {
		_, err := c.Predict(audio)
		require.NoError(t, err)
	}
	runtime.GC()
	runtime.ReadMemStats(&m)

	// Allow some allocation growth but not unbounded
	// Each iteration allocates ~100KB for scores/predictions, so 100 iters ≈ 10MB expected
	// If we see >100MB growth, tensors are probably leaking
	growth := m.TotalAlloc - baseAlloc
	assert.Less(t, growth, uint64(100*1024*1024), "memory grew by %d MB, possible tensor leak", growth/(1024*1024))
}

func TestClassifier_LabelCountMismatch(t *testing.T) {
	modelPath := os.Getenv("BIRDNET_V24_MODEL")
	if modelPath == "" {
		t.Skip("BIRDNET_V24_MODEL not set")
	}

	// Provide wrong number of labels
	_, err := NewClassifier(modelPath, WithLabels([]string{"a", "b", "c"}))
	require.Error(t, err)
	var labelErr *LabelCountError
	require.ErrorAs(t, err, &labelErr)
}

func TestClassifierAndRangeFilter_FullPipeline(t *testing.T) {
	modelPath := os.Getenv("BIRDNET_V24_MODEL")
	labelsPath := os.Getenv("BIRDNET_V24_LABELS")
	metaPath := os.Getenv("BIRDNET_META_MODEL")
	if modelPath == "" || labelsPath == "" || metaPath == "" {
		t.Skip("BIRDNET_V24_MODEL, BIRDNET_V24_LABELS, or BIRDNET_META_MODEL not set")
	}

	// Create classifier
	c, err := NewClassifier(modelPath, WithLabelsPath(labelsPath), WithTopK(10))
	require.NoError(t, err)
	defer c.Close()

	// Create range filter using classifier's labels
	rf, err := NewRangeFilter(metaPath,
		WithRangeFilterFromClassifierLabels(c.Labels()),
		WithRangeFilterThreshold(0.03),
	)
	require.NoError(t, err)
	defer rf.Close()

	// Predict with silence
	audio := make([]float32, c.Config().SampleCount)
	result, err := c.Predict(audio)
	require.NoError(t, err)

	// Get location scores for Helsinki, June 15
	scores, err := rf.Predict(60.17, 24.94, 6, 15)
	require.NoError(t, err)

	// Filter predictions
	filtered := rf.Filter(result.Predictions, scores, true)
	// Filtered should have equal or fewer predictions
	assert.LessOrEqual(t, len(filtered), len(result.Predictions))
}

func TestRangeFilter_Integration(t *testing.T) {
	metaPath := os.Getenv("BIRDNET_META_MODEL")
	labelsPath := os.Getenv("BIRDNET_V24_LABELS")
	if metaPath == "" || labelsPath == "" {
		t.Skip("BIRDNET_META_MODEL or BIRDNET_V24_LABELS not set")
	}

	rf, err := NewRangeFilter(metaPath, WithRangeFilterLabelsPath(labelsPath))
	require.NoError(t, err)
	defer rf.Close()

	// Helsinki, June 15
	scores, err := rf.Predict(60.17, 24.94, 6, 15)
	require.NoError(t, err)
	assert.NotEmpty(t, scores)

	// At least some species should have non-zero scores
	nonZero := 0
	for _, s := range scores {
		if s.Score > 0.01 {
			nonZero++
		}
	}
	assert.Greater(t, nonZero, 0, "expected some species with non-zero scores for Helsinki in June")
}
```

- [ ] **Step 2: Verify unit tests still pass (integration tests skipped by default)**

Run: `cd birdnet && go test ./... -v`
Expected: all unit tests PASS, integration tests not compiled

- [ ] **Step 3: Commit**

```bash
git add birdnet/classifier_test.go
git commit -m "feat: add integration tests for classifier and range filter"
```

---

### Task 11: Final Verification and Cleanup

- [ ] **Step 1: Run all unit tests**

Run: `cd birdnet && go test ./... -v -count=1`
Expected: all PASS

- [ ] **Step 2: Run go vet**

Run: `cd birdnet && go vet ./...`
Expected: no issues

- [ ] **Step 3: Run golangci-lint**

Run: `cd birdnet && golangci-lint run ./...`
Expected: no errors (warnings acceptable)

- [ ] **Step 4: Fix any lint issues found in step 3**

- [ ] **Step 5: Format code**

Run: `cd birdnet && gofmt -w .`

- [ ] **Step 6: Final commit**

```bash
git add -A birdnet/
git commit -m "chore: fix lint issues and format code"
```

---

## Task Dependency Graph

```
Task 1 (scaffold)
  ├─> Task 2 (types)
  │     ├─> Task 3 (errors)
  │     ├─> Task 4 (postprocess) ──────────────┐
  │     ├─> Task 5 (labels) ───────────────────┐│
  │     └─> Task 6 (detection) ───────────────┐││
  │                                            │││
  ├─> Task 7 (rangefilter pure logic) ────────┐│││
  │                                           ││││
  └─> Task 8 (classifier) <── depends on 2-6 ─┘│││
      └─> Task 9 (rangefilter ONNX) <── 7,8 ──┘││
          └─> Task 10 (integration tests) <─────┘│
              └─> Task 11 (final cleanup) <───────┘
```

Tasks 2, 3, 4, 5, 6, 7 are independent of each other and can be parallelized.
Task 8 depends on 2-6. Task 9 depends on 7 and 8. Task 10 depends on 9.
