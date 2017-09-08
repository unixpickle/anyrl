package anypg

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/serializer"
)

// Default settings for TRPO.
const (
	DefaultTargetKL        = 0.01
	DefaultLineSearchDecay = 0.7
	DefaultMaxLineSearch   = 20
)

// TRPO uses the Trust Region Policy Optimization
// algorithm to train agents.
//
// See https://arxiv.org/abs/1502.05477.
type TRPO struct {
	NaturalPG

	// TarketKL specifies the desired average KL
	// divergence after taking a step.
	// It is used to scale gradients.
	//
	// If 0, DefaultTargetKL is used.
	TargetKL float64

	// LineSearchDecay is an exponential decay factor
	// used to decay the step size until TargetKL is
	// satisfied and the approximate loss has improved.
	// It should be between 0 and 1.
	//
	// If 0, DefaultLineSearchDecay is used.
	LineSearchDecay float64

	// MaxLineSearch is the maximum number of line-search
	// iterations to take.
	// This usually won't matter, as long as it is high
	// enough.
	//
	// If 0, DefaultMaxLineSearch is used.
	MaxLineSearch int

	// LogLineSearch is called after each iteration of the
	// objective line search.
	//
	// If nil, no logging is done.
	LogLineSearch func(meanKL, meanImprovement anyvec.Numeric)
}

// Run computes a step to improve the agent's performance
// on the rollouts.
func (t *TRPO) Run(r *anyrl.RolloutSet) anydiff.Grad {
	res := t.NaturalPG.run(r)
	if res.ZeroGrad {
		return res.Grad
	}
	c := r.Creator()
	stepSize := t.stepSize(res)

	res.Grad.Scale(stepSize)

	for i := 0; i < t.maxLineSearch(); i++ {
		if t.acceptable(r, res) {
			break
		}
		res.Grad.Scale(c.MakeNumeric(t.lineSearchDecay()))
	}

	return res.Grad
}

func (t *TRPO) stepSize(r *naturalPGRes) anyvec.Numeric {
	c := r.Creator()
	ops := c.NumOps()
	r.ReducedOut.Reuse()
	dotProd := dotGrad(r.Grad, t.applyFisher(r.ReducedRollouts, r.Grad, r.ReducedOut))
	zero := c.MakeNumeric(0)

	// The fisher-vector product might be less than zero due
	// to rounding errors.
	if ops.Less(dotProd, zero) || ops.Equal(dotProd, zero) {
		return zero
	}

	return ops.Pow(
		ops.Div(c.MakeNumeric(2*t.targetKL()), dotProd),
		c.MakeNumeric(0.5),
	)
}

func (t *TRPO) acceptable(r *anyrl.RolloutSet, npg *naturalPGRes) bool {
	c := npg.Creator()
	inSeq := lazyseq.TapeRereader(r.Inputs)
	rewardSeq := lazyseq.TapeRereader(t.actionJudger().JudgeActions(r).Tape(c))
	newOutSeq := t.apply(inSeq, t.steppedPolicy(npg.Grad))
	sampledOut := lazyseq.TapeRereader(r.Actions)
	npg.PolicyOut.Reuse()

	// At each timestep, compute a pair <improvement, kl divergence>.
	mappedOut := lazyseq.MapN(func(n int, v ...anydiff.Res) anydiff.Res {
		reward := v[0]
		oldOut := v[1]
		newOut := v[2]
		sampled := v[3].Output()

		// Importance sampling
		probRatio := anydiff.Exp(anydiff.Sub(
			t.ActionSpace.LogProb(newOut, sampled, n),
			t.ActionSpace.LogProb(oldOut, sampled, n),
		))

		rewardChange := anydiff.Sub(anydiff.Mul(probRatio, reward), reward)
		kl := t.ActionSpace.KL(oldOut, newOut, n)

		// Put the rewards and kl divergences side-by-side.
		joined := c.Concat(rewardChange.Output(), kl.Output())
		transposed := c.MakeVector(joined.Len())
		anyvec.Transpose(joined, transposed, 2)

		return anydiff.NewConst(transposed)
	}, rewardSeq, npg.PolicyOut, newOutSeq, sampledOut)

	outStats := lazyseq.Mean(mappedOut).Output()
	improvement := anyvec.Sum(outStats.Slice(0, 1))
	kl := anyvec.Sum(outStats.Slice(1, 2))

	if t.LogLineSearch != nil {
		t.LogLineSearch(kl, improvement)
	}

	targetImprovement := c.MakeNumeric(0)
	targetKL := c.MakeNumeric(t.targetKL())
	ops := c.NumOps()
	return ops.Greater(improvement, targetImprovement) &&
		ops.Less(kl, targetKL)
}

func (t *TRPO) steppedPolicy(step anydiff.Grad) anyrnn.Block {
	copied, err := serializer.Copy(t.Policy)
	if err != nil {
		panic(err)
	}
	newParams := anynet.AllParameters(copied)
	oldParams := anynet.AllParameters(t.Policy)
	newGrad := anydiff.Grad{}
	for i, old := range oldParams {
		if gradVal, ok := step[old]; ok {
			newGrad[newParams[i]] = gradVal
		}
	}
	if len(newGrad) != len(step) {
		panic("not all parameters are visible to anynet.AllParameters")
	}
	newGrad.AddToVars()
	return copied.(anyrnn.Block)
}

func (t *TRPO) targetKL() float64 {
	if t.TargetKL == 0 {
		return DefaultTargetKL
	} else {
		return t.TargetKL
	}
}

func (t *TRPO) lineSearchDecay() float64 {
	if t.LineSearchDecay == 0 {
		return DefaultLineSearchDecay
	} else {
		return t.LineSearchDecay
	}
}

func (t *TRPO) maxLineSearch() int {
	if t.MaxLineSearch == 0 {
		return DefaultMaxLineSearch
	} else {
		return t.MaxLineSearch
	}
}

func (t *TRPO) actionJudger() ActionJudger {
	if t.ActionJudger == nil {
		return &TotalJudger{Normalize: true}
	} else {
		return t.ActionJudger
	}
}
