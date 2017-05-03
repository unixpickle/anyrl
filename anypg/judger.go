package anypg

import (
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
)

// An ActionJudger generates a signal indicating how
// "good" actions were in a set of rollouts.
// An ActionJudger is used to decide which
// actions to encourage and which ones to discourage
// in policy gradient methods.
//
// The JudgeActions method produces a tape of the same
// dimensions as the tape of rewards.
type ActionJudger interface {
	JudgeActions(rollouts *anyrl.RolloutSet) lazyseq.Tape
}

// QJudger is an ActionJudger which judges the goodness of
// an action by that action's sampled Q-value.
type QJudger struct{}

// JudgeActions transforms the rewards so that each reward
// is replaced with the sum of all the rewards from that
// timestep to the end of the episode.
func (q QJudger) JudgeActions(r *anyrl.RolloutSet) lazyseq.Tape {
	sum := anyrl.TotalRewardsBatch(r.Rewards)

	resTape, writer := lazyseq.ReferenceTape()

	for batch := range r.Rewards.ReadTape(0, -1) {
		// Create two separate copies of the sum to
		// avoid modifying the batch we send.
		writer <- sum.Reduce(batch.Present)
		sum = sum.Reduce(batch.Present)
		sum.Packed.Sub(batch.Packed)
	}

	close(writer)
	return resTape
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
func (t *TotalJudger) JudgeActions(r *anyrl.RolloutSet) lazyseq.Tape {
	sum := anyrl.TotalRewardsBatch(r.Rewards)
	if sum == nil {
		return r.Rewards
	}

	if t.Normalize {
		t.normalize(sum.Packed)
	}

	resTape, writer := lazyseq.ReferenceTape()
	for batch := range r.Rewards.ReadTape(0, -1) {
		writer <- sum.Reduce(batch.Present)
	}
	close(writer)
	return resTape
}

func (t *TotalJudger) normalize(vec anyvec.Vector) {
	c := vec.Creator()

	// Due to rounding errors, homogeneous rewards (i.e.
	// all the same reward) sometimes yield vectors of
	// the form <-1, -1, -1, ...>.
	// By subtracting the maximum value, we can ensure
	// zeroes for homogeneous vectors.
	max := anyvec.Max(vec)
	vec.AddScalar(c.NumOps().Mul(max, c.MakeNumeric(-1)))

	// Set mean=0 so we can use second moment as variance.
	meanVals := vec.Copy()
	meanVals.Scale(c.MakeNumeric(-1 / float64(vec.Len())))
	vec.AddScalar(anyvec.Sum(meanVals))

	stdVals := vec.Copy()

	epsilon := t.Epsilon
	if epsilon == 0 {
		epsilon = 1e-8
	}
	stdVals.AddScalar(c.MakeNumeric(epsilon))

	anyvec.Pow(stdVals, c.MakeNumeric(2))
	stdVals.Scale(c.MakeNumeric(1 / float64(vec.Len())))
	secondMoment := anyvec.Sum(stdVals)
	invStdVec := c.MakeVector(1)
	invStdVec.AddScalar(secondMoment)
	anyvec.Pow(invStdVec, c.MakeNumeric(-0.5))
	invStd := anyvec.Sum(invStdVec)

	vec.Scale(invStd)
}
