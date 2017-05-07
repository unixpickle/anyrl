package anyrl

import (
	"math"
	"testing"
)

func TestRewardVariance(t *testing.T) {
	r := Rewards([][]float64{
		{1, 2, 3, 1},
		{2, -1},
		{-1, -1, -2},
	})

	// Sums: 7 1 -4; mean=1.33333; var=20.222...
	variance := r.Variance()
	expected := 182.0 / 9
	if math.Abs(variance-expected) > 1e-3 {
		t.Errorf("bad variance: %v (expected %v)", variance, expected)
	}
}
