package anypg

import (
	"github.com/unixpickle/anydiff"
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

// PG implements vanilla policy gradients.
type PG struct {
	// Policy applies the policy to a sequence of inputs.
	Policy func(s lazyseq.Rereader) lazyseq.Rereader

	// Params specifies which parameters to include in
	// the gradients.
	Params []*anydiff.Var

	// ActionSpace determines log-likelihoods of actions.
	ActionSpace anyrl.LogProber

	// ActionJudger is used to judge actions.
	//
	// If nil, TotalJudger is used.
	ActionJudger ActionJudger

	// Set these to enable entropy regularization.
	//
	// A term is added to every timestep of the form
	// EntropyReg*H(policy(state)).
	Entropyer  anyrl.Entropyer
	EntropyReg float64
}

// Run performs policy gradients on the rollouts.
func (p *PG) Run(r *anyrl.RolloutSet) anydiff.Grad {
	grad := anydiff.NewGrad(p.Params...)
	if len(grad) == 0 {
		return grad
	}
	c := creatorFromGrad(grad)

	policyOut := p.Policy(lazyseq.TapeRereader(c, r.Inputs))

	selectedOuts := lazyseq.TapeRereader(c, r.SampledOuts)
	rewards := lazyseq.TapeRereader(c, p.actionJudger().JudgeActions(r))

	scores := lazyseq.MapN(func(n int, v ...anydiff.Res) anydiff.Res {
		actionParams := v[0]
		selected := v[1]
		rewards := v[2]
		logProb := p.ActionSpace.LogProb(actionParams, selected.Output(), n)
		cost := anydiff.Mul(logProb, rewards)
		if p.Entropyer != nil && p.EntropyReg != 0 {
			entropy := p.Entropyer.Entropy(actionParams, n)
			cost = anydiff.Add(cost, anydiff.Scale(entropy,
				c.MakeNumeric(p.EntropyReg)))
		}
		return cost
	}, policyOut, selectedOuts, rewards)

	score := lazyseq.Mean(scores)
	one := c.MakeVector(1)
	one.AddScalar(c.MakeNumeric(1))
	score.Propagate(one, grad)

	return grad
}

func (p *PG) actionJudger() ActionJudger {
	if p.ActionJudger == nil {
		return &TotalJudger{Normalize: true}
	} else {
		return p.ActionJudger
	}
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

func creatorFromGrad(g anydiff.Grad) anyvec.Creator {
	for _, v := range g {
		return v.Creator()
	}
	panic("empty gradient")
}
