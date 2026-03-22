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
	assert.InDelta(t, 3.0, BirdNETv24.Duration(), 1e-6)
	assert.InDelta(t, 5.0, BirdNETv30.Duration(), 1e-6)
	assert.InDelta(t, 5.0, PerchV2.Duration(), 1e-6)
}

func TestModelType_SampleCount(t *testing.T) {
	assert.Equal(t, 144000, BirdNETv24.SampleCount())
	assert.Equal(t, 160000, BirdNETv30.SampleCount())
	assert.Equal(t, 160000, PerchV2.SampleCount())
}
