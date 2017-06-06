package anyes

import "math"

// MeanReward computes the mean reward for some rollouts.
func MeanReward(r []*Rollout) float64 {
	var sum float64
	for _, x := range r {
		sum += x.Reward
	}
	return sum / float64(len(r))
}

// normalize adjusts the values to have mean 0 and
// variance 1.
func normalize(vals []float64) {
	var mean float64
	allEqual := true
	for _, x := range vals {
		if x != vals[0] {
			allEqual = false
		}
		mean += x
	}

	if allEqual {
		for i := range vals {
			vals[i] = 0
		}
		return
	}

	mean /= float64(len(vals))

	var variance float64
	for _, x := range vals {
		diff := x - mean
		variance += diff * diff
	}
	variance /= float64(len(vals))

	scale := 1 / math.Sqrt(variance)
	for i, x := range vals {
		vals[i] = (x - mean) * scale
	}
}
