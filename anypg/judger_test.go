package anypg

import (
	"math"
	"testing"

	"github.com/unixpickle/anyrl"
)

func TestQJudger(t *testing.T) {
	rewards := [][]float64{
		{1, 0.5, 2},
		{},
		{0.5, -1},
	}

	actual := (&QJudger{}).JudgeActions(&anyrl.RolloutSet{Rewards: rewards})
	expected := [][]float64{
		{3.5, 2.5, 2},
		{},
		{-0.5, -1},
	}

	testRewardsEquiv(t, actual, expected)
}

func TestQJudgerDiscounted(t *testing.T) {
	rewards := [][]float64{
		{1, 0.5, 2},
		{},
		{0.5, -1},
	}
	j := &QJudger{Discount: 0.5}

	actual := j.JudgeActions(&anyrl.RolloutSet{Rewards: rewards})
	expected := [][]float64{
		{1.75, 1.5, 2},
		{},
		{0, -1},
	}

	testRewardsEquiv(t, actual, expected)
}

func TestTotalJudger(t *testing.T) {
	rewards := [][]float64{
		{1, 2, 3, 1},
		{2, -1},
		{-1, -1, -2},
	}

	// Sums: 7 1 -4; mean=1.33333; std=4.4969
	// Normalized:  1.260131  -0.074125  -1.186005

	judger := &TotalJudger{Normalize: true}

	actual := judger.JudgeActions(&anyrl.RolloutSet{Rewards: rewards})
	expected := [][]float64{
		{1.260131, 1.260131, 1.260131, 1.260131},
		{-0.074125, -0.074125},
		{-1.186005, -1.186005, -1.186005},
	}

	testRewardsEquiv(t, actual, expected)
}

func testRewardsEquiv(t *testing.T, actual, expected anyrl.Rewards) {
	if len(actual) != len(expected) {
		t.Errorf("expected %d sequences but got %d", len(expected), len(actual))
		return
	}
	for seqIdx, actualSeq := range actual {
		expectedSeq := expected[seqIdx]
		match := true
		if len(actualSeq) != len(expectedSeq) {
			match = false
		} else {
			for i, x := range expectedSeq {
				a := actualSeq[i]
				if math.Abs(x-a) > 1e-4 {
					match = false
					break
				}
			}
		}
		if !match {
			t.Errorf("sequence %d: expected %v but got %v", seqIdx,
				expectedSeq, actualSeq)
		}
	}
}
