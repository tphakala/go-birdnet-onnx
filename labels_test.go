package birdnet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadLabels_Text(t *testing.T) {
	labels, err := loadLabels("testdata/labels.txt")
	require.NoError(t, err)
	require.Len(t, labels, 3)
	assert.Equal(t, "Turdus merula_Common Blackbird", labels[0])
	assert.Equal(t, "Parus major_Great Tit", labels[1])
	assert.Equal(t, "Erithacus rubecula_European Robin", labels[2])
}

func TestLoadLabels_Text_EmptyLines(t *testing.T) {
	content := "\n  Turdus merula  \n\n  Parus major  \n\n"
	labels, err := loadLabelsFromBytes([]byte(content), ".txt")
	require.NoError(t, err)
	require.Len(t, labels, 2)
	assert.Equal(t, "Turdus merula", labels[0])
	assert.Equal(t, "Parus major", labels[1])
}

func TestLoadLabels_CSV_Semicolon(t *testing.T) {
	labels, err := loadLabels("testdata/labels_semicolon.csv")
	require.NoError(t, err)
	require.Len(t, labels, 3)
	assert.Equal(t, "Turdus merula", labels[0])
	assert.Equal(t, "Parus major", labels[1])
}

func TestLoadLabels_CSV_Comma(t *testing.T) {
	labels, err := loadLabels("testdata/labels_comma.csv")
	require.NoError(t, err)
	require.Len(t, labels, 3)
	assert.Equal(t, "Turdus merula", labels[0])
}

func TestLoadLabels_JSON_Array(t *testing.T) {
	labels, err := loadLabels("testdata/labels_array.json")
	require.NoError(t, err)
	require.Len(t, labels, 3)
	assert.Equal(t, "Turdus merula", labels[0])
}

func TestLoadLabels_JSON_Object(t *testing.T) {
	labels, err := loadLabels("testdata/labels_object.json")
	require.NoError(t, err)
	require.Len(t, labels, 3)
	assert.Equal(t, "Turdus merula", labels[0])
}

func TestLoadLabels_JSON_Named(t *testing.T) {
	labels, err := loadLabels("testdata/labels_named.json")
	require.NoError(t, err)
	require.Len(t, labels, 3)
	assert.Equal(t, "Turdus merula", labels[0])
}

func TestLoadLabels_FileNotFound(t *testing.T) {
	_, err := loadLabels("testdata/nonexistent.txt")
	require.Error(t, err)
	var labelErr *LabelLoadError
	require.ErrorAs(t, err, &labelErr)
	assert.Contains(t, labelErr.Path, "nonexistent.txt")
}

func TestLoadLabels_UnsupportedExtension(t *testing.T) {
	_, err := loadLabelsFromBytes([]byte("test"), ".xyz")
	require.Error(t, err)
	var labelErr *LabelLoadError
	require.ErrorAs(t, err, &labelErr)
	assert.Contains(t, labelErr.Reason, "unsupported")
}
