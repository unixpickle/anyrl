package anyrl

import (
	"fmt"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyrnn"
)

// Default target KL divergence for TRPO.
const DefaultTargetKL = 0.01

// Default line-search decay factor for TRPO.
const DefaultLineSearchDecay = 0.5

// Default maximum number of line-search iterations for
// TRPO.
const DefaultMaxLineSearch = 20

// TRPO uses the Trust Region Policy Optimization
// algorithm to train agents.
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
}

// Run computes a step to improve the agent's performance
// on the rollouts.
//
// This may temporarily modify the policy.
// Thus, it is not safe to use Run() while using the
// policy on a different Goroutine.
func (t *TRPO) Run(r *RolloutSet) anydiff.Grad {
	grad := t.NaturalPG.Run(r)
	if len(grad) == 0 {
		return grad
	}
	c := creatorFromGrad(grad)
	outs := t.storePolicyOutputs(c, r)
	stepSize := t.stepSize(r, grad, outs)

	grad.Scale(stepSize)

	for i := 0; i < t.maxLineSearch(); i++ {
		if t.acceptable(r, grad, outs) {
			break
		}
		grad.Scale(c.MakeNumeric(t.lineSearchDecay()))
	}

	return grad
}

func (t *TRPO) stepSize(r *RolloutSet, grad anydiff.Grad, outs lazyrnn.Tape) anyvec.Numeric {
	c := creatorFromGrad(grad)
	dotProd := dotGrad(grad, t.applyFisher(r, grad, outs))
	resVec := c.MakeVector(1)
	resVec.AddScalar(dotProd)
	anyvec.Pow(resVec, c.MakeNumeric(-1))
	resVec.Scale(c.MakeNumeric(2 * t.targetKL()))
	anyvec.Pow(resVec, c.MakeNumeric(0.5))
	return anyvec.Sum(resVec)
}

func (t *TRPO) acceptable(r *RolloutSet, grad anydiff.Grad, outs lazyrnn.Tape) bool {
	backup := t.backupParams()
	grad.AddToVars()
	defer t.restoreParams(backup)

	c := creatorFromGrad(grad)
	inSeq := lazyrnn.TapeRereader(c, r.Inputs)
	rewardSeq := lazyrnn.TapeRereader(c, r.RemainingRewards())
	oldOutSeq := lazyrnn.TapeRereader(c, outs)
	newOutSeq := t.apply(inSeq, t.Policy)
	sampledOut := lazyrnn.TapeRereader(c, r.SampledOuts)

	// At each timestep, compute a pair <improvement, kl divergence>.
	mappedOut := lazyrnn.MapN(func(n int, v ...anydiff.Res) anydiff.Res {
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
	}, rewardSeq, oldOutSeq, newOutSeq, sampledOut)

	outStats := lazyrnn.Mean(mappedOut).Output()
	improvement := anyvec.Sum(outStats.Slice(0, 1))
	kl := anyvec.Sum(outStats.Slice(1, 2))

	switch val := improvement.(type) {
	case float32:
		if val < 0 || float64(kl.(float32)) > t.targetKL() {
			return false
		}
	case float64:
		if val < 0 || kl.(float64) > t.targetKL() {
			return false
		}
	default:
		panic(fmt.Sprintf("unsupported numeric: %T", val))
	}

	return true
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

func (t *TRPO) backupParams() []anyvec.Vector {
	var res []anyvec.Vector
	for _, p := range t.Params {
		res = append(res, p.Vector.Copy())
	}
	return res
}

func (t *TRPO) restoreParams(backup []anyvec.Vector) {
	for i, x := range backup {
		t.Params[i].Vector.Set(x)
	}
}
