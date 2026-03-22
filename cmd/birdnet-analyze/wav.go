package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

type wavFormat struct {
	audioFormat   uint16
	numChannels   uint16
	sampleRate    uint32
	bitsPerSample uint16
}

// readWAV reads a mono 16-bit PCM WAV file and returns normalized float32 samples.
func readWAV(path string) (samples []float32, sampleRate int, err error) {
	f, err := os.Open(path) //nolint:gosec // Path from CLI argument
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = f.Close() }()

	if err := validateRIFFHeader(f); err != nil {
		return nil, 0, err
	}

	format, dataSize, err := readChunks(f)
	if err != nil {
		return nil, 0, err
	}

	if err := validateFormat(format); err != nil {
		return nil, 0, err
	}

	samples, err = readPCMData(f, dataSize)
	if err != nil {
		return nil, 0, err
	}

	return samples, int(format.sampleRate), nil
}

func validateRIFFHeader(r io.Reader) error {
	var header [12]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return fmt.Errorf("failed to read RIFF header: %w", err)
	}
	if string(header[:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return fmt.Errorf("not a valid WAV file")
	}
	return nil
}

func readChunks(f io.ReadSeeker) (*wavFormat, uint32, error) {
	var format *wavFormat
	for {
		id, size, err := readChunkHeader(f)
		if err != nil {
			break
		}

		switch id {
		case "fmt ":
			format, err = readFmtChunk(f, size)
			if err != nil {
				return nil, 0, err
			}
		case "data":
			if format == nil {
				return nil, 0, fmt.Errorf("data chunk found before fmt chunk")
			}
			return format, size, nil
		default:
			if err := skipChunk(f, size); err != nil {
				return nil, 0, err
			}
		}
	}
	return nil, 0, fmt.Errorf("missing fmt or data chunk")
}

func readChunkHeader(r io.Reader) (id string, size uint32, err error) {
	var raw [4]byte
	if _, err = io.ReadFull(r, raw[:]); err != nil {
		return "", 0, err
	}
	if err := binary.Read(r, binary.LittleEndian, &size); err != nil {
		return "", 0, err
	}
	return string(raw[:]), size, nil
}

func skipChunk(f io.ReadSeeker, size uint32) error {
	skipSize := int64(size)
	if skipSize%2 != 0 {
		skipSize++
	}
	_, err := f.Seek(skipSize, io.SeekCurrent)
	return err
}

func readFmtChunk(r io.ReadSeeker, chunkSize uint32) (*wavFormat, error) {
	if chunkSize < 16 {
		return nil, fmt.Errorf("fmt chunk too small: %d bytes", chunkSize)
	}

	var wf wavFormat
	if err := binary.Read(r, binary.LittleEndian, &wf.audioFormat); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &wf.numChannels); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &wf.sampleRate); err != nil {
		return nil, err
	}
	// Skip byte rate (4) and block align (2)
	var skip [6]byte
	if _, err := io.ReadFull(r, skip[:]); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &wf.bitsPerSample); err != nil {
		return nil, err
	}
	// Skip any extra fmt bytes
	if chunkSize > 16 {
		if _, err := r.Seek(int64(chunkSize-16), io.SeekCurrent); err != nil {
			return nil, err
		}
	}
	return &wf, nil
}

func validateFormat(f *wavFormat) error {
	if f.audioFormat != 1 {
		return fmt.Errorf("unsupported audio format %d (expected PCM = 1)", f.audioFormat)
	}
	if f.numChannels != 1 {
		return fmt.Errorf("expected mono audio (1 channel), got %d channels", f.numChannels)
	}
	if f.bitsPerSample != 16 {
		return fmt.Errorf("expected 16-bit audio, got %d-bit", f.bitsPerSample)
	}
	return nil
}

func readPCMData(r io.Reader, dataSize uint32) ([]float32, error) {
	numSamples := int(dataSize) / 2
	raw := make([]int16, numSamples)
	if err := binary.Read(r, binary.LittleEndian, raw); err != nil {
		return nil, fmt.Errorf("failed to read audio data: %w", err)
	}

	samples := make([]float32, numSamples)
	for i, s := range raw {
		samples[i] = float32(s) / 32768.0
	}
	return samples, nil
}
