package anypg

import (
	"math"

	"github.com/unixpickle/anyrl"
)

// An ActionJudger generates a signal indicating how
// "good" actions were in a set of rollouts.
// An ActionJudger is used to decide which actions
// to encourage and which ones to discourage.
//
// The JudgeActions method produces an anyrl.Rewards of
// the same dimensions as the original reward signal.
type ActionJudger interface {
	JudgeActions(rollouts *anyrl.RolloutSet) anyrl.Rewards
}

// QJudger is an ActionJudger which judges the goodness of
// an action by that action's sampled Q-value.
type QJudger struct {
	// Discount is the reward discount factor.
	//
	// If 0, no discount is used.
	Discount float64
}

// JudgeActions transforms the rewards so that each reward
// is replaced with the sum of all the rewards from that
// timestep to the end of the episode.
func (q *QJudger) JudgeActions(r *anyrl.RolloutSet) anyrl.Rewards {
	var res anyrl.Rewards
	for _, seq := range r.Rewards {
		newSeq := make([]float64, len(seq))
		var sum float64
		for t := len(seq) - 1; t >= 0; t-- {
			if q.Discount != 0 {
				sum *= q.Discount
			}
			sum += seq[t]
			newSeq[t] = sum
		}
		res = append(res, newSeq)
	}
	return res
}

// TotalJudger is an ActionJudger which judges the
// goodness of an episode by the total reward of the
// encompassing episode.
type TotalJudger struct {
	// Normalize, if true, tells the judger to process
	// the total rewards to have a mean of zero and a
	// standard deviation of 1.
	Normalize bool

	// Epsilon is a small fudge factor used to prevent
	// numerical issues when dividing by the standard
	// deviation.
	// It is only needed if Normalize is true.
	//
	// If this is 0, a reasonably small value is used.
	Epsilon float64
}

// JudgeActions repeats the cumulative rewards at every
// timestep in a tape.
func (t *TotalJudger) JudgeActions(r *anyrl.RolloutSet) anyrl.Rewards {
	totals := r.Rewards.Totals()

	if t.Normalize {
		t.normalize(totals)
	}

	var res anyrl.Rewards
	for seqIdx, seq := range r.Rewards {
		var newSeq []float64
		for _ = range seq {
			newSeq = append(newSeq, totals[seqIdx])
		}
		res = append(res, newSeq)
	}

	return res
}

func (t *TotalJudger) normalize(rewards []float64) {
	var heterogeneous bool
	var sum float64

	for _, x := range rewards {
		sum += x
		if x != rewards[0] {
			heterogeneous = true
		}
	}
	if !heterogeneous {
		for i := range rewards {
			rewards[i] = 0
		}
		return
	}

	mean := sum / float64(len(rewards))

	var sqSum float64
	for i := range rewards {
		rewards[i] -= mean
		sqSum += rewards[i] * rewards[i]
	}
	variance := sqSum / float64(len(rewards))

	epsilon := t.Epsilon
	if epsilon == 0 {
		epsilon = 1e-8
	}

	normalizer := 1 / (math.Sqrt(variance) + epsilon)
	for i := range rewards {
		rewards[i] *= normalizer
	}
}
