package birdnet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCalculateWeek_January1(t *testing.T) {
	assert.InDelta(t, float32(1.0), CalculateWeek(1, 1), 1e-6)
}

func TestCalculateWeek_January8(t *testing.T) {
	assert.InDelta(t, float32(2.0), CalculateWeek(1, 8), 1e-6)
}

func TestCalculateWeek_January28(t *testing.T) {
	assert.InDelta(t, float32(4.0), CalculateWeek(1, 28), 1e-6)
}

func TestCalculateWeek_January31_Clamped(t *testing.T) {
	assert.InDelta(t, float32(4.0), CalculateWeek(1, 31), 1e-6)
}

func TestCalculateWeek_February1(t *testing.T) {
	assert.InDelta(t, float32(5.0), CalculateWeek(2, 1), 1e-6)
}

func TestCalculateWeek_December31_Clamped(t *testing.T) {
	assert.InDelta(t, float32(48.0), CalculateWeek(12, 31), 1e-6)
}

func TestCalculateWeek_December1(t *testing.T) {
	assert.InDelta(t, float32(45.0), CalculateWeek(12, 1), 1e-6)
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
		{Species: "species_b", Score: 0.01, Index: 1},
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
	require.Len(t, result[0], 1)
	require.Empty(t, result[1])
}
