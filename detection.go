package birdnet

import "fmt"

func detectModelTypeFromShapes(inputShapes [][]int64, numOutputs int) (ModelType, error) {
	if len(inputShapes) == 0 {
		return 0, &ModelDetectionError{Reason: "model has no input tensors"}
	}

	shape := inputShapes[0]
	if len(shape) < 2 {
		return 0, &ModelDetectionError{Reason: fmt.Sprintf("input shape has %d dimensions, expected at least 2", len(shape))}
	}

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

func buildModelConfig(mt ModelType, inputShape []int64, numOutputs int) ModelConfig {
	cfg := ModelConfig{
		Type:           mt,
		SampleRate:     mt.SampleRate(),
		Duration:       mt.Duration(),
		SampleCount:    mt.SampleCount(),
		NumOutputs:     numOutputs,
		EmbeddingIndex: -1,
		InputShape:     make([]int64, len(inputShape)),
	}
	copy(cfg.InputShape, inputShape)

	switch mt {
	case BirdNETv24:
		cfg.LogitsIndex = 0
		cfg.EmbeddingSize = 0
	case BirdNETv30:
		cfg.LogitsIndex = 1
		cfg.EmbeddingIndex = 0
		cfg.EmbeddingSize = 1280
	case PerchV2:
		cfg.LogitsIndex = 3
		cfg.EmbeddingIndex = 0
		cfg.EmbeddingSize = 1536
	}

	return cfg
}
