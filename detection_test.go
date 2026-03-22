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
	cfg := buildModelConfig(BirdNETv24, []int64{1, 144000}, 1)
	assert.Equal(t, BirdNETv24, cfg.Type)
	assert.Equal(t, 48000, cfg.SampleRate)
	assert.InDelta(t, 3.0, cfg.Duration, 1e-6)
	assert.Equal(t, 144000, cfg.SampleCount)
	assert.Equal(t, 0, cfg.EmbeddingSize)
	assert.Equal(t, -1, cfg.EmbeddingIndex)
	assert.Equal(t, 0, cfg.LogitsIndex)
}

func TestBuildModelConfig_BirdNETv30(t *testing.T) {
	cfg := buildModelConfig(BirdNETv30, []int64{1, 160000}, 2)
	assert.Equal(t, 1280, cfg.EmbeddingSize)
	assert.Equal(t, 0, cfg.EmbeddingIndex)
	assert.Equal(t, 1, cfg.LogitsIndex)
}

func TestBuildModelConfig_PerchV2(t *testing.T) {
	cfg := buildModelConfig(PerchV2, []int64{1, 160000}, 4)
	assert.Equal(t, 1536, cfg.EmbeddingSize)
	assert.Equal(t, 0, cfg.EmbeddingIndex)
	assert.Equal(t, 3, cfg.LogitsIndex)
}
