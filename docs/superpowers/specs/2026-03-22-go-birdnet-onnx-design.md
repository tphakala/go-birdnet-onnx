# Go BirdNET ONNX Library Design

A Go library for running inference on BirdNET and Google Perch bird species classification models using ONNX Runtime. Port of the Rust `birdnet-onnx` library.

## Scope

**V1 includes:**
- Classifier supporting BirdNET v2.4, BirdNET v3.0, and Perch v2 models
- RangeFilter (location/date-based species filtering via BirdNET meta model)
- CPU and XNNPACK execution providers (primary target: Raspberry Pi / ARM)
- Label loading from text, CSV, and JSON formats
- Model type auto-detection from ONNX tensor shapes

**V1 excludes (deferred):**
- BSG Finland model support and post-processing
- CUDA/TensorRT dedicated config structs (not relevant for target hardware)
- BatchInferenceContext (GPU memory reuse — no GPU)
- Timeout/cancellation (can be added later via `RunOptions.Terminate()`)

## Dependencies

- `github.com/yalue/onnxruntime_go` — ONNX Runtime Go bindings (v1.24+)
- Go 1.26 standard library only (no third-party dependencies beyond onnxruntime_go)

## Architecture

Single flat package (`birdnet`). All public types and functions at the top level. No sub-packages.

### Model Specifications

| Property | BirdNET v2.4 | BirdNET v3.0 | Perch v2 |
|---|---|---|---|
| Sample Rate | 48,000 Hz | 32,000 Hz | 32,000 Hz |
| Duration | 3.0s | 5.0s | 5.0s |
| Sample Count | 144,000 | 160,000 | 160,000 |
| Input Shape | [batch, 144000] | [batch, 160000] | [batch, 160000] |
| Output Tensors | 1 (logits) | 2 (embeddings, logits) | 4 (embedding, spatial, spectrogram, logits) |
| Activation | Sigmoid | Sigmoid | Softmax |
| Species Count | ~6,522 | ~11,560 | ~14,795 |
| Embedding Size | none | 1,280 | 1,536 |

### Input Shape Flexibility

Models accept either `[batch, samples]` or `[batch, 1, samples]`. The library detects the actual input shape from the ONNX model metadata and reshapes accordingly. This is important because different model exports may use either format.

## Types

### ModelType

```go
type ModelType int

const (
    BirdNETv24 ModelType = iota
    BirdNETv30
    PerchV2
)

func (m ModelType) String() string
func (m ModelType) SampleRate() int
func (m ModelType) Duration() float64
func (m ModelType) SampleCount() int
```

Methods on `ModelType` return the fixed parameters for each model. This replaces the need for a separate `ModelConfig` lookup — the enum itself carries the knowledge.

### ModelConfig

```go
type ModelConfig struct {
    Type          ModelType
    SampleRate    int
    Duration      float64
    SampleCount   int
    NumOutputs    int
    EmbeddingSize int  // 0 if model doesn't produce embeddings
    LogitsIndex   int  // which output tensor contains logits
    InputShape    []int64 // actual shape from ONNX model (e.g., [1, 144000] or [1, 1, 144000])
}
```

Constructed internally during model loading. Exposed read-only via `Classifier.Config()`.

### Prediction and Result

```go
type Prediction struct {
    Species    string
    Confidence float32
    Index      int
}

type Result struct {
    ModelType   ModelType
    Predictions []Prediction // top-K, sorted descending by confidence
    Embeddings  []float32    // nil if model doesn't produce embeddings
    RawScores   []float32    // all scores after activation (sigmoid/softmax)
}
```

### LocationScore

```go
type LocationScore struct {
    Species string
    Score   float32
    Index   int
}
```

## Classifier

### Construction

```go
func NewClassifier(modelPath string, opts ...ClassifierOption) (*Classifier, error)
```

Functional options pattern:

```go
type ClassifierOption func(*classifierConfig)

func WithModelType(t ModelType) ClassifierOption
func WithLabels(labels []string) ClassifierOption
func WithLabelsPath(path string) ClassifierOption
func WithTopK(k int) ClassifierOption                      // default: 10
func WithMinConfidence(threshold float32) ClassifierOption  // default: 0.0
func WithSessionOptions(fn func(*ort.SessionOptions)) ClassifierOption
```

`WithSessionOptions` takes a callback that receives the `*ort.SessionOptions` before session creation. This lets the caller configure any execution provider without the library needing to enumerate them:

```go
classifier, err := birdnet.NewClassifier("model.onnx",
    birdnet.WithLabelsPath("labels.txt"),
    birdnet.WithTopK(5),
    birdnet.WithSessionOptions(func(opts *ort.SessionOptions) {
        opts.AppendExecutionProvider("XNNPACK", map[string]string{
            "intra_op_num_threads": "4",
        })
    }),
)
```

#### Construction Sequence

1. Load ONNX model metadata via `ort.GetInputOutputInfo(modelPath)` to read input/output tensor names and shapes.
2. Auto-detect `ModelType` from shapes (or use override from `WithModelType`).
3. Build `ModelConfig` from detected type + actual ONNX shapes.
4. Load labels (from `WithLabels`, `WithLabelsPath`, or error if none provided).
5. Validate label count matches model output size.
6. Create `*ort.SessionOptions` with defaults (`IntraOpNumThreads(1)`, `InterOpNumThreads(1)`).
7. Apply user's `WithSessionOptions` callback if provided.
8. Create `*ort.DynamicAdvancedSession` with input/output names.
9. Destroy `SessionOptions` immediately (session doesn't take ownership).
10. Return `*Classifier`.

### Audio Input Requirements

The `Predict` and `PredictBatch` methods accept raw audio samples as `[]float32`:

- **Mono channel only.** Stereo or multi-channel audio must be mixed to mono before passing to the library.
- **Correct sample rate.** Audio must be resampled to the model's sample rate (48kHz for BirdNET v2.4, 32kHz for v3.0 and Perch). The library does not resample.
- **Normalized to [-1.0, 1.0].** Standard float audio range. For 16-bit PCM: `sample / 32768.0`.
- **Exact segment length.** The slice length must equal `Config().SampleCount` exactly (144,000 or 160,000 samples). An `InputSizeError` is returned otherwise.

Audio loading, resampling, and segmentation are the caller's responsibility. This library is inference-only.

### Inference

```go
func (c *Classifier) Predict(audio []float32) (*Result, error)
func (c *Classifier) PredictBatch(segments [][]float32) ([]*Result, error)
func (c *Classifier) Config() ModelConfig
func (c *Classifier) Labels() []string
func (c *Classifier) Close() error
```

#### Predict Flow

1. Validate `len(audio) == c.config.SampleCount`.
2. Create input tensor: `ort.NewTensor[float32](inputShape, audio)` where `inputShape` is `[1, sampleCount]` or `[1, 1, sampleCount]` depending on what the model expects. The onnxruntime_go library wraps the Go slice directly (no copy).
3. Create pre-sized output tensors using `ort.NewEmptyTensor[float32](shape)` with known shapes from the model config. Output shapes are fixed per model type (e.g., `[1, 6522]` for BirdNET v2.4 logits). For models with multiple outputs (v3.0, Perch), create one tensor per output.
4. Call `session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensors...})`.
5. Extract logits from the correct output tensor (`config.LogitsIndex`) via `GetData()`.
6. Apply activation: sigmoid for BirdNET models, softmax for Perch.
7. Extract embeddings if available.
8. Select top-K predictions above `minConfidence`, sorted descending.
9. Destroy all tensors (input and all outputs) via `defer`.
10. Return `*Result`.

**Important:** All tensor `Destroy()` calls use `defer` immediately after creation to ensure cleanup even on error paths. This prevents CGo memory leaks.

#### PredictBatch Flow

1. Return error if `len(segments) == 0` (zero-batch causes ONNX Runtime errors).
2. Validate all segments have correct length.
3. Concatenate segments into a single flat `[]float32` slice.
4. Create input tensor with shape `[batchSize, sampleCount]` (or `[batchSize, 1, sampleCount]`).
5. Run inference once.
6. Split output logits by batch dimension.
7. Apply activation and top-K per segment.
8. Return `[]*Result`.

### Tensor Lifecycle

Each `Predict`/`PredictBatch` call:
- Creates input tensor wrapping the caller's audio slice (no copy — onnxruntime_go uses the Go slice directly).
- Creates pre-sized output tensors with known shapes from model config.
- All tensors are `defer`-destroyed immediately after creation, ensuring cleanup even on error paths.
- Reads output data via `GetData()` into Go-owned slices before tensors are destroyed.

No tensors escape the method boundary. No possibility of tensor leaks.

**Performance note:** Tensor creation/destruction involves CGo calls but is negligible compared to model inference time (~100-500ms per segment on Raspberry Pi). For high-throughput scenarios, a future optimization could pool tensors via `sync.Pool`, but this is not needed for v1.

### Thread Safety

`DynamicAdvancedSession` is safe for concurrent `Run()` calls. Multiple goroutines can call `Predict()` on the same `*Classifier` concurrently. Each call creates its own tensors.

Default session options set `IntraOpNumThreads(1)` and `InterOpNumThreads(1)` to prevent ONNX Runtime's internal thread pool from contending with Go's goroutine scheduler. On Raspberry Pi this is the right default — one inference at a time per core.

## Model Auto-Detection

```go
func detectModelType(info *ort.InputOutputInfo) (ModelType, error)
```

Internal function. Logic:

1. Extract input tensor shape. Determine sample count from the last dimension.
2. Extract number of output tensors.
3. Match:
   - 144,000 samples + 1 output → `BirdNETv24`
   - 160,000 samples + 2 outputs → `BirdNETv30`
   - 160,000 samples + 4 outputs → `PerchV2`
   - Otherwise → error with descriptive message

This mirrors the Rust `detection.rs` logic. The user can override with `WithModelType()` (needed if a future model has the same shape signature as an existing one).

## Label Loading

```go
func loadLabels(path string) ([]string, error)
```

Internal function. Detects format by file extension and content:

### Text Format (`.txt`)
One label per line. Whitespace trimmed. Empty lines skipped.

### CSV Format (`.csv`)
Auto-detects delimiter (comma or semicolon). Intelligent column selection by header name:
- Priority: `sci_name` > `com_name` > `name` > `label`
- Falls back to first non-numeric column if no recognized header
- Handles optional numeric index column

Uses `encoding/csv` from the standard library.

### JSON Format (`.json`)
Supports three structures:
- `["label1", "label2"]` — flat array
- `{"labels": ["label1", "label2"]}` — object with labels key
- `[{"name": "label1"}, ...]` — array of objects

Uses `encoding/json` from the standard library.

### Validation
After loading, label count is validated against the model's logits output dimension. Mismatch produces a descriptive error.

## Post-Processing

### Sigmoid (BirdNET v2.4, v3.0)

```go
func sigmoid(x float32) float32 {
    return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}
```

Applied element-wise to the logits tensor. Each score is independent (multi-label classification).

### Softmax (Perch v2)

```go
func softmax(logits []float32) []float32
```

Standard numerically-stable softmax: subtract max, exponentiate, normalize. Scores sum to 1.0 (single-label classification).

### Top-K Selection

```go
func topK(scores []float32, labels []string, k int, minConf float32) []Prediction
```

1. Pair each score with its label and index.
2. Filter by `minConf`.
3. Partial sort to find top-K (use `slices.SortFunc` — no need for a heap with typical K=10 and N=6500-15000).
4. Return sorted descending by confidence.

## RangeFilter

The BirdNET meta model predicts species occurrence probability based on geographic location and date.

### Construction

```go
func NewRangeFilter(modelPath string, opts ...RangeFilterOption) (*RangeFilter, error)

type RangeFilterOption func(*rangeFilterConfig)

func WithRangeFilterLabels(labels []string) RangeFilterOption
func WithRangeFilterLabelsPath(path string) RangeFilterOption
func WithRangeFilterThreshold(threshold float32) RangeFilterOption
func WithRangeFilterFromClassifierLabels(labels []string) RangeFilterOption
```

The meta model takes a 3-element input tensor with shape `[1, 3]` containing `[latitude, longitude, week]` and outputs species occurrence probabilities as a tensor with shape `[1, numSpecies]`.

### API

```go
func (r *RangeFilter) Predict(latitude, longitude float32, month, day int) ([]LocationScore, error)
func (r *RangeFilter) Filter(predictions []Prediction, scores []LocationScore, rerank bool) []Prediction
func (r *RangeFilter) FilterBatch(batches [][]Prediction, scores []LocationScore, rerank bool) [][]Prediction
func (r *RangeFilter) Close() error
```

### Week Calculation

BirdNET uses a 48-week calendar (4 weeks per month, 7 days per week):

```go
func CalculateWeek(month, day int) float32
```

Formula:
```
weeksFromMonths = (month - 1) * 4
weekInMonth = min((day - 1) / 7 + 1, 4)
week = weeksFromMonths + weekInMonth
```

The result is clamped to [1, 48]. BirdNET expects exactly 4 weeks per month (48-week year). Days 29-31 are treated as belonging to week 4 of their month.

Examples: Jan 1 → week 1, Jan 8 → week 2, Jan 28 → week 4, Jan 31 → week 4, Feb 1 → week 5, Dec 31 → week 48.

**Note:** This differs from the Rust library which does not clamp. The Go implementation corrects this to match BirdNET's expected 48-week encoding.

### Coordinate and Date Validation

```go
func ValidateCoordinates(latitude, longitude float32) error
func ValidateDate(month, day int) error
```

Latitude: [-90, 90]. Longitude: [-180, 180]. Month: [1, 12]. Day: [1, 31].

### Filter Logic

`Filter` removes predictions for species with location scores below the threshold. If `rerank` is true, each prediction's confidence is multiplied by its location score, and the result is re-sorted.

`FilterBatch` applies the same logic to multiple prediction sets (from `PredictBatch`).

## Error Handling

Using Go 1.26's `errors.AsType` for ergonomic error type checking. Sentinel errors where appropriate, typed errors for structured context.

```go
var (
    ErrModelPathRequired = errors.New("birdnet: model path is required")
    ErrLabelsRequired    = errors.New("birdnet: labels are required")
)

type InputSizeError struct {
    Expected int
    Got      int
}

type BatchInputSizeError struct {
    Index    int
    Expected int
    Got      int
}

type LabelCountError struct {
    Expected int
    Got      int
}

type ModelDetectionError struct {
    Reason string
}

type LabelLoadError struct {
    Path   string
    Reason string
}

type InvalidCoordinatesError struct {
    Latitude  float32
    Longitude float32
    Reason    string
}

type InvalidDateError struct {
    Month  int
    Day    int
    Reason string
}
```

All error types implement `error` via `Error() string` methods. The `fmt.Errorf("...: %w", err)` pattern is used for wrapping ONNX Runtime errors.

Callers can use `errors.AsType[*InputSizeError](err)` (Go 1.26) for type-safe error inspection.

## ONNX Runtime Initialization

The library does NOT call `ort.InitializeEnvironment()` itself. The caller is responsible for initializing and destroying the ONNX Runtime environment. This is because:

1. Only one global ORT environment can exist at a time.
2. The caller needs to control the library path (`ort.SetSharedLibraryPath`).
3. The caller may have other ONNX models beyond bird classification.

Convenience helpers are provided:

```go
// MustInitORT initializes the ONNX Runtime with the given library path.
// Panics on failure. Intended for simple applications.
func MustInitORT(libraryPath string)

// DestroyORT tears down the ONNX Runtime environment.
func DestroyORT()
```

## File Structure

```
birdnet/
    birdnet.go          # package doc, MustInitORT, DestroyORT
    classifier.go       # Classifier, ClassifierOption, NewClassifier
    detection.go        # detectModelType, ModelConfig construction
    errors.go           # error types and sentinel errors
    labels.go           # loadLabels (text, CSV, JSON parsing)
    postprocess.go      # sigmoid, softmax, topK
    rangefilter.go      # RangeFilter, RangeFilterOption, NewRangeFilter
    types.go            # ModelType, ModelConfig, Prediction, Result, LocationScore
    birdnet_test.go     # unit tests for internal functions
    classifier_test.go  # classifier tests (with test fixtures)
    detection_test.go   # model detection tests
    labels_test.go      # label parsing tests
    postprocess_test.go # post-processing tests
    rangefilter_test.go # range filter tests
```

All in one package: `package birdnet`.

## Testing Strategy

### Unit Tests (no ONNX models needed)

- **Label parsing**: text, CSV (comma and semicolon), JSON (all three formats), edge cases (empty lines, Unicode, missing headers)
- **Post-processing**: sigmoid correctness, softmax correctness and numerical stability, top-K selection, min confidence filtering
- **Model detection**: shape-to-ModelType mapping for all model types, error cases
- **Week calculation**: known month/day → week mappings, boundary values
- **Coordinate/date validation**: valid and invalid inputs
- **Error formatting**: all error types produce readable messages

### Integration Tests (require ONNX model files)

- **Classifier construction**: load each model type, verify config
- **Single inference**: known audio → expected top species
- **Batch inference**: multiple segments, verify independent results
- **Input validation**: wrong sample count → `InputSizeError`
- **Label validation**: wrong count → `LabelCountError`
- **RangeFilter**: known location/date → expected species filtering
- **RangeFilter + Classifier**: full pipeline
- **Concurrency**: multiple goroutines calling `Predict` on the same `Classifier` — verify no panics or data corruption
- **Memory stability**: loop test running thousands of `Predict` calls, asserting RSS stays bounded (catches missed `tensor.Destroy()` on error paths)

Integration tests gated behind `//go:build integration` build tag to avoid failures without model files.

## Usage Example

```go
package main

import (
    "fmt"
    "log"

    ort "github.com/yalue/onnxruntime_go"
    "myproject/birdnet"
)

func main() {
    // Initialize ONNX Runtime
    ort.SetSharedLibraryPath("/usr/lib/libonnxruntime.so.1.24.4")
    if err := ort.InitializeEnvironment(); err != nil {
        log.Fatal(err)
    }
    defer ort.DestroyEnvironment()

    // Create classifier (model type auto-detected, IntraOp/InterOp threads default to 1)
    classifier, err := birdnet.NewClassifier("BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx",
        birdnet.WithLabelsPath("labels.txt"),
        birdnet.WithTopK(5),
        birdnet.WithMinConfidence(0.1),
    )
    if err != nil {
        log.Fatal(err)
    }
    defer classifier.Close()

    // Load and segment your audio (not part of this library)
    audio := loadAudioSegment("recording.wav", classifier.Config().SampleRate, classifier.Config().SampleCount)

    // Run inference
    result, err := classifier.Predict(audio)
    if err != nil {
        log.Fatal(err)
    }

    for _, p := range result.Predictions {
        fmt.Printf("%-40s %.2f%%\n", p.Species, p.Confidence*100)
    }

    // Optional: filter by location
    filter, err := birdnet.NewRangeFilter("BirdNET_GLOBAL_6K_V2.4_MData_Model_FP16.onnx",
        birdnet.WithRangeFilterFromClassifierLabels(classifier.Labels()),
    )
    if err != nil {
        log.Fatal(err)
    }
    defer filter.Close()

    scores, err := filter.Predict(60.17, 24.94, 6, 15) // Helsinki, June 15
    if err != nil {
        log.Fatal(err)
    }

    filtered := filter.Filter(result.Predictions, scores, true)
    for _, p := range filtered {
        fmt.Printf("%-40s %.2f%% (location-adjusted)\n", p.Species, p.Confidence*100)
    }
}
```

## Design Decisions Summary

| Decision | Rationale |
|---|---|
| Functional options | Idiomatic Go, extensible without breaking API |
| `DynamicAdvancedSession` | Thread-safe concurrent inference without mutex |
| Single package | Simple, portable, easy to copy into other projects |
| Caller-managed ORT init | Only one global ORT environment allowed; caller controls lifecycle |
| `IntraOpNumThreads(1)` default | Prevents ORT internal threading from contending with Go scheduler (critical on RPi) |
| Pre-sized output tensors | Known shapes from model config; avoids auto-allocation ownership issues |
| No timeout/cancellation in v1 | Adds complexity; RPi inference is fast enough; can add via `RunOptions` later |
| `errors.AsType` (Go 1.26) | Type-safe error inspection without boilerplate |
| `//go:build integration` for model tests | Unit tests run without model files; integration tests opt-in |
| `WithSessionOptions(func)` callback | Provider-agnostic; any EP works without library changes |
