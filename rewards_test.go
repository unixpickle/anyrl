package anyrl

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/lazyseq"
)

func TestRewardVariance(t *testing.T) {
	c := anyvec64.DefaultCreator{}

	tapeIn, writer := lazyseq.ReferenceTape()
	writer <- &anyseq.Batch{
		Present: []bool{true, false, true, true},
		Packed:  c.MakeVectorData([]float64{1, 2, -1}),
	}
	writer <- &anyseq.Batch{
		Present: []bool{true, false, true, true},
		Packed:  c.MakeVectorData([]float64{2, -1, -1}),
	}
	writer <- &anyseq.Batch{
		Present: []bool{true, false, false, true},
		Packed:  c.MakeVectorData([]float64{3, -2}),
	}
	writer <- &anyseq.Batch{
		Present: []bool{true, false, false, false},
		Packed:  c.MakeVectorData([]float64{1}),
	}
	close(writer)

	// Sums: 7 1 -4; mean=1.33333; var=20.222...
	variance := RewardVariance(c, tapeIn)
	expected := 182.0 / 9
	if math.Abs(variance.(float64)-expected) > 1e-3 {
		t.Errorf("bad variance: %v (expected %v)", variance, expected)
	}
}
