package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	birdnet "github.com/tphakala/go-birdnet-onnx"
)

type config struct {
	audioPath  string
	modelPath  string
	labelsPath string
	ortLib     string
	overlap    float64
	topK       int
	minConf    float64
	modelType  string
	batchSize  int
	csvPath    string
	verbose    bool
}

func main() {
	cfg := parseFlags()

	if err := run(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func parseFlags() *config {
	cfg := &config{}
	flag.StringVar(&cfg.modelPath, "model", "", "path to ONNX model file (required)")
	flag.StringVar(&cfg.labelsPath, "labels", "", "path to labels file (required)")
	flag.StringVar(&cfg.ortLib, "ort-lib", "", "path to ONNX Runtime shared library (required)")
	flag.Float64Var(&cfg.overlap, "overlap", 0.0, "overlap between segments in seconds")
	flag.IntVar(&cfg.topK, "top-k", 3, "number of top predictions per segment")
	flag.Float64Var(&cfg.minConf, "min-confidence", 0.1, "minimum confidence threshold")
	flag.StringVar(&cfg.modelType, "model-type", "", "override model type detection (v24, v30, perch)")
	flag.IntVar(&cfg.batchSize, "batch-size", 8, "batch size for inference")
	flag.StringVar(&cfg.csvPath, "csv", "", "output results to CSV file")
	flag.BoolVar(&cfg.verbose, "verbose", false, "enable verbose logging")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: birdnet-analyze [flags] <audio.wav>\n\n")
		fmt.Fprintf(os.Stderr, "Analyze bird sounds in a WAV file using BirdNET/Perch ONNX models.\n\n")
		fmt.Fprintf(os.Stderr, "Arguments:\n")
		fmt.Fprintf(os.Stderr, "  audio.wav    Input WAV file (mono, 16-bit PCM)\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}

	flag.Parse()

	if flag.NArg() < 1 || cfg.modelPath == "" || cfg.labelsPath == "" || cfg.ortLib == "" {
		flag.Usage()
		os.Exit(1)
	}
	cfg.audioPath = flag.Arg(0)
	return cfg
}

func run(cfg *config) error {
	if cfg.verbose {
		fmt.Fprintf(os.Stderr, "Initializing ONNX Runtime from %s\n", cfg.ortLib)
	}
	initStart := time.Now()
	birdnet.MustInitORT(cfg.ortLib)
	defer birdnet.DestroyORT()
	if cfg.verbose {
		fmt.Fprintf(os.Stderr, "ONNX Runtime initialized in %s\n", time.Since(initStart))
	}

	classifier, err := createClassifier(cfg)
	if err != nil {
		return err
	}
	defer func() { _ = classifier.Close() }()

	modelCfg := classifier.Config()

	samples, sampleRate, err := loadAudio(cfg, modelCfg.SampleRate)
	if err != nil {
		return err
	}

	audioDuration := float64(len(samples)) / float64(sampleRate)
	segments, offsets := chunkAudio(samples, modelCfg.SampleCount, cfg.overlap, sampleRate)

	output("Model: %s (%.1fs segments, %.1fs overlap)\n", modelCfg.Type, modelCfg.Duration, cfg.overlap)
	output("Analyzing: %s (%s, %d Hz)\n", cfg.audioPath, formatDuration(audioDuration), sampleRate)
	output("Segments: %d, Batch size: %d\n\n", len(segments), cfg.batchSize)

	csvWriter, csvCloser, err := openCSV(cfg.csvPath)
	if err != nil {
		return err
	}
	if csvCloser != nil {
		defer csvCloser()
	}

	totalSegments, err := runInference(classifier, segments, offsets, &modelCfg, cfg.batchSize, csvWriter)
	if err != nil {
		return err
	}

	printSummary(totalSegments, audioDuration, modelCfg.Duration, cfg.csvPath)
	return nil
}

func createClassifier(cfg *config) (*birdnet.Classifier, error) {
	opts := []birdnet.ClassifierOption{
		birdnet.WithLabelsPath(cfg.labelsPath),
		birdnet.WithTopK(cfg.topK),
		birdnet.WithMinConfidence(float32(cfg.minConf)),
	}

	if cfg.modelType != "" {
		mt, err := parseModelType(cfg.modelType)
		if err != nil {
			return nil, err
		}
		opts = append(opts, birdnet.WithModelType(mt))
	}

	if cfg.verbose {
		fmt.Fprintf(os.Stderr, "Loading model from %s\n", cfg.modelPath)
	}
	modelStart := time.Now()
	classifier, err := birdnet.NewClassifier(cfg.modelPath, opts...)
	if err != nil {
		return nil, fmt.Errorf("creating classifier: %w", err)
	}
	if cfg.verbose {
		fmt.Fprintf(os.Stderr, "Model loaded in %s\n", time.Since(modelStart))
	}
	return classifier, nil
}

func loadAudio(cfg *config, expectedRate int) (samples []float32, sampleRate int, err error) {
	if cfg.verbose {
		fmt.Fprintf(os.Stderr, "Reading %s\n", cfg.audioPath)
	}
	readStart := time.Now()
	samples, sampleRate, err = readWAV(cfg.audioPath)
	if err != nil {
		return nil, 0, fmt.Errorf("reading WAV: %w", err)
	}
	if cfg.verbose {
		fmt.Fprintf(os.Stderr, "Read %d samples in %s\n", len(samples), time.Since(readStart))
	}
	if sampleRate != expectedRate {
		return nil, 0, fmt.Errorf("WAV sample rate %d does not match model sample rate %d", sampleRate, expectedRate)
	}
	return samples, sampleRate, nil
}

func openCSV(path string) (*csv.Writer, func(), error) {
	if path == "" {
		return nil, nil, nil
	}
	f, err := os.Create(path) //nolint:gosec // Path from CLI flag
	if err != nil {
		return nil, nil, fmt.Errorf("creating CSV file: %w", err)
	}
	w := csv.NewWriter(f)
	closer := func() {
		w.Flush()
		_ = f.Close()
	}
	if err := w.Write([]string{"Start (s)", "End (s)", "Scientific name", "Common name", "Confidence"}); err != nil {
		_ = f.Close()
		return nil, nil, fmt.Errorf("writing CSV header: %w", err)
	}
	return w, closer, nil
}

var inferStart time.Time

func runInference(
	classifier *birdnet.Classifier,
	segments [][]float32, offsets []float64,
	modelCfg *birdnet.ModelConfig, batchSize int,
	csvWriter *csv.Writer,
) (int, error) {
	inferStart = time.Now()
	totalSegments := 0

	for batchStart := 0; batchStart < len(segments); batchStart += batchSize {
		batchEnd := min(batchStart+batchSize, len(segments))
		batch := segments[batchStart:batchEnd]
		batchOffsets := offsets[batchStart:batchEnd]

		results, err := inferBatch(classifier, batch)
		if err != nil {
			return 0, err
		}

		for i, result := range results {
			offset := batchOffsets[i]
			totalSegments++

			if len(result.Predictions) == 0 {
				continue
			}

			preds := make([]string, len(result.Predictions))
			for j, p := range result.Predictions {
				preds[j] = fmt.Sprintf("%s (%.1f%%)", p.Species, p.Confidence*100)
			}
			output("%s  %s\n", formatTimestamp(offset), strings.Join(preds, ", "))

			if csvWriter != nil {
				endTime := offset + modelCfg.Duration
				if err := writeCSVResults(csvWriter, result.Predictions, offset, endTime); err != nil {
					return 0, err
				}
			}
		}
	}
	return totalSegments, nil
}

func inferBatch(classifier *birdnet.Classifier, batch [][]float32) ([]*birdnet.Result, error) {
	if len(batch) == 1 {
		result, err := classifier.Predict(batch[0])
		if err != nil {
			return nil, fmt.Errorf("inference failed: %w", err)
		}
		return []*birdnet.Result{result}, nil
	}
	results, err := classifier.PredictBatch(batch)
	if err != nil {
		return nil, fmt.Errorf("batch inference failed: %w", err)
	}
	return results, nil
}

func writeCSVResults(w *csv.Writer, predictions []birdnet.Prediction, start, end float64) error {
	for _, p := range predictions {
		sci, common := splitSpeciesLabel(p.Species)
		if err := w.Write([]string{
			fmt.Sprintf("%.1f", start),
			fmt.Sprintf("%.1f", end),
			sci,
			common,
			fmt.Sprintf("%.4f", p.Confidence),
		}); err != nil {
			return fmt.Errorf("writing CSV row: %w", err)
		}
	}
	return nil
}

func printSummary(totalSegments int, audioDuration, segDuration float64, csvPath string) {
	elapsed := time.Since(inferStart)
	segPerSec := float64(totalSegments) / elapsed.Seconds()
	audioPerSec := float64(totalSegments) * segDuration / elapsed.Seconds()

	output("\n%d segments of %s audio analyzed in %s (%.1f segments/s, %.1fx realtime)\n",
		totalSegments, formatDuration(audioDuration), elapsed.Round(time.Millisecond), segPerSec, audioPerSec)

	if csvPath != "" {
		output("Results written to %s\n", csvPath)
	}
}

// output writes formatted text to stdout, ignoring write errors.
func output(format string, args ...any) {
	_, _ = fmt.Fprintf(os.Stdout, format, args...)
}

func parseModelType(s string) (birdnet.ModelType, error) {
	switch strings.ToLower(s) {
	case "v24", "birdnetv24":
		return birdnet.BirdNETv24, nil
	case "v30", "birdnetv30":
		return birdnet.BirdNETv30, nil
	case "perch", "perchv2":
		return birdnet.PerchV2, nil
	default:
		return 0, fmt.Errorf("unknown model type %q (valid: v24, v30, perch)", s)
	}
}

func chunkAudio(samples []float32, segmentSize int, overlapSec float64, sampleRate int) (segments [][]float32, offsets []float64) {
	overlapSamples := int(overlapSec * float64(sampleRate))
	step := max(segmentSize-overlapSamples, 1)
	capacity := len(samples)/step + 1
	segments = make([][]float32, 0, capacity)
	offsets = make([]float64, 0, capacity)

	for pos := 0; pos < len(samples); pos += step {
		segment := make([]float32, segmentSize)
		end := min(pos+segmentSize, len(samples))
		copy(segment, samples[pos:end])
		segments = append(segments, segment)
		offsets = append(offsets, float64(pos)/float64(sampleRate))
	}

	return segments, offsets
}

func formatTimestamp(seconds float64) string {
	mins := int(seconds) / 60
	secs := seconds - float64(mins*60)
	return fmt.Sprintf("%02d:%04.1f", mins, secs)
}

func formatDuration(seconds float64) string {
	if seconds < 60 {
		return fmt.Sprintf("%.1fs", seconds)
	}
	mins := int(seconds) / 60
	secs := seconds - float64(mins*60)
	if mins < 60 {
		return fmt.Sprintf("%dm %.0fs", mins, secs)
	}
	hours := mins / 60
	mins %= 60
	return fmt.Sprintf("%dh %dm", hours, mins)
}

func splitSpeciesLabel(label string) (scientific, common string) {
	parts := strings.SplitN(label, "_", 2)
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	return label, ""
}
