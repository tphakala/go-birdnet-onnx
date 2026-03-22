//go:build integration

package birdnet

import (
	"os"
	"runtime"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	ort "github.com/yalue/onnxruntime_go"
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

	growth := m.TotalAlloc - baseAlloc
	assert.Less(t, growth, uint64(100*1024*1024), "memory grew by %d MB, possible tensor leak", growth/(1024*1024))
}

func TestClassifier_LabelCountMismatch(t *testing.T) {
	modelPath := os.Getenv("BIRDNET_V24_MODEL")
	if modelPath == "" {
		t.Skip("BIRDNET_V24_MODEL not set")
	}

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

	c, err := NewClassifier(modelPath, WithLabelsPath(labelsPath), WithTopK(10))
	require.NoError(t, err)
	defer c.Close()

	rf, err := NewRangeFilter(metaPath,
		WithRangeFilterFromClassifierLabels(c.Labels()),
		WithRangeFilterThreshold(0.03),
	)
	require.NoError(t, err)
	defer rf.Close()

	audio := make([]float32, c.Config().SampleCount)
	result, err := c.Predict(audio)
	require.NoError(t, err)

	scores, err := rf.Predict(60.17, 24.94, 6, 15)
	require.NoError(t, err)

	filtered := rf.Filter(result.Predictions, scores, true)
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

	scores, err := rf.Predict(60.17, 24.94, 6, 15)
	require.NoError(t, err)
	assert.NotEmpty(t, scores)

	nonZero := 0
	for _, s := range scores {
		if s.Score > 0.01 {
			nonZero++
		}
	}
	assert.Greater(t, nonZero, 0, "expected some species with non-zero scores for Helsinki in June")
}
