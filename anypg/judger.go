package anypg

import (
	"math"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
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

	// Normalize, if true, indicates that the resulting
	// Q-values should be statistically normalized.
	Normalize bool

	// Epsilon is a small fudge factor used to prevent
	// numerical issues when dividing by the standard
	// deviation.
	// It is only needed if Normalize is true.
	//
	// If this is 0, a reasonably small value is used.
	Epsilon float64
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

	if q.Normalize {
		normalized := flattenRewards(res)
		tj := &TotalJudger{Normalize: q.Normalize, Epsilon: q.Epsilon}
		tj.normalize(normalized)
		unflattenRewards(res, normalized)
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

// A GAEJudger uses Generalized Advantage Estimation to
// judge actions based on the predictions from a value
// estimator.
//
// For more on GAE, see: https://arxiv.org/abs/1506.02438.
type GAEJudger struct {
	// ValueFunc takes a batch of observation sequences
	// and produces a batch of value sequences.
	// It can assume that the resulting channel will be
	// fully read by the caller.
	ValueFunc func(inputs lazyseq.Rereader) <-chan *anyseq.Batch

	// Discount is the reward discount factor.
	// Values closer to 1 give a longer time horizon.
	Discount float64

	// Lambda ranges from 0 to 1 and controls the amount of
	// variance (0 = low variance).
	Lambda float64
}

// JudgeActions computes generalized advantage estimates.
func (g *GAEJudger) JudgeActions(r *anyrl.RolloutSet) anyrl.Rewards {
	input := lazyseq.TapeRereader(r.Inputs)
	criticOut := g.ValueFunc(input)

	estimatedValues := make([][]float64, len(r.Rewards))
	for outBatch := range criticOut {
		comps := vectorToComponents(outBatch.Packed)
		for i, pres := range outBatch.Present {
			if pres {
				estimatedValues[i] = append(estimatedValues[i], comps[0])
				comps = comps[1:]
			}
		}
	}

	var res [][]float64
	for i, rewSeq := range r.Rewards {
		valSeq := estimatedValues[i]
		advantages := make([]float64, len(rewSeq))
		var accumulation float64
		for t := len(rewSeq) - 1; t >= 0; t-- {
			delta := rewSeq[t] - valSeq[t]
			if t+1 < len(rewSeq) {
				delta += g.Discount * valSeq[t+1]
			}
			accumulation *= g.Discount * g.Lambda
			accumulation += delta
			advantages[t] = accumulation
		}
		res = append(res, advantages)
	}

	return anyrl.Rewards(res)
}

func flattenRewards(r anyrl.Rewards) []float64 {
	var values []float64
	for _, seq := range r {
		for _, x := range seq {
			values = append(values, x)
		}
	}
	return values
}

func unflattenRewards(output anyrl.Rewards, flat []float64) {
	for _, seq := range output {
		for i := range seq {
			seq[i] = flat[0]
			flat = flat[1:]
		}
	}
}

func vectorToComponents(vec anyvec.Vector) []float64 {
	switch data := vec.Data().(type) {
	case []float32:
		res := make([]float64, len(data))
		for i, x := range data {
			res[i] = float64(x)
		}
		return res
	case []float64:
		return data
	default:
		panic("unsupported numeric type")
	}
}
