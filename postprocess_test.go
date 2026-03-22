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
		{10, 0.99995},
		{-10, 0.0000454},
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
	assert.Equal(t, float32(0), input[0])
}

func TestSoftmax(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0}
	result := softmax(input)
	require.Len(t, result, 3)
	var sum float32
	for _, v := range result {
		sum += v
	}
	assert.InDelta(t, 1.0, sum, 0.0001)
	assert.Greater(t, result[2], result[1])
	assert.Greater(t, result[1], result[0])
}

func TestSoftmax_NumericalStability(t *testing.T) {
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
