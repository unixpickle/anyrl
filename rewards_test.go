package anyrl

import (
	"math"
	"reflect"
	"testing"

	"github.com/unixpickle/anyvec/anyvec64"
)

func TestRewardsTape(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	r := Rewards([][]float64{
		{1, 2, 3},
		nil,
		{5, 4, 3, 2, 1},
		{3, 2, 1},
		nil,
		{2},
	})
	tape := r.Tape(c)

	reconstructed := make(Rewards, len(r))
	for step := range tape.ReadTape(0, -1) {
		if len(step.Present) != len(r) {
			t.Fatalf("length should be %d but got %d", len(r), len(step.Present))
		}
		vec := step.Packed.Data().([]float64)
		for i, p := range step.Present {
			if p {
				reconstructed[i] = append(reconstructed[i], vec[0])
				vec = vec[1:]
			}
		}
	}

	if !reflect.DeepEqual([][]float64(r), [][]float64(reconstructed)) {
		t.Errorf("expected %v but got %v", r, reconstructed)
	}
}

func TestRewardsVariance(t *testing.T) {
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

func TestRewardsReduce(t *testing.T) {
	r := Rewards([][]float64{
		{1, 2, 3, 1},
		{2, -1},
		{-1, -1, -2},
		{1, 2, 5},
	})
	actual := r.Reduce([]bool{false, true, false, true})
	expected := Rewards([][]float64{
		nil,
		{2, -1},
		nil,
		{1, 2, 5},
	})
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}
