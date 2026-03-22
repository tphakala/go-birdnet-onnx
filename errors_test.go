package birdnet

import (
	"errors"
	"fmt"
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
	require.ErrorIs(t, fmt.Errorf("wrap: %w", ErrModelPathRequired), ErrModelPathRequired)
	require.ErrorIs(t, fmt.Errorf("wrap: %w", ErrLabelsRequired), ErrLabelsRequired)
}
