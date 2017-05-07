package anypg

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
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
	DefaultLineSearchDecay = 0.5
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
}

// Run computes a step to improve the agent's performance
// on the rollouts.
func (t *TRPO) Run(r *anyrl.RolloutSet) anydiff.Grad {
	grad := t.NaturalPG.Run(r)
	if len(grad) == 0 || allZeros(grad) {
		return grad
	}
	c := creatorFromGrad(grad)
	outs, outSeq := t.storePolicyOutputs(c, r)
	outSeq.Reuse()
	stepSize := t.stepSize(r, grad, outSeq)

	grad.Scale(stepSize)

	for i := 0; i < t.maxLineSearch(); i++ {
		if t.acceptable(r, grad, outs) {
			break
		}
		grad.Scale(c.MakeNumeric(t.lineSearchDecay()))
	}

	return grad
}

func (t *TRPO) stepSize(r *anyrl.RolloutSet, grad anydiff.Grad,
	outSeq lazyseq.Rereader) anyvec.Numeric {
	c := creatorFromGrad(grad)
	dotProd := dotGrad(grad, t.applyFisher(r, grad, outSeq))
	resVec := c.MakeVector(1)
	resVec.AddScalar(dotProd)
	anyvec.Pow(resVec, c.MakeNumeric(-1))
	resVec.Scale(c.MakeNumeric(2 * t.targetKL()))
	anyvec.Pow(resVec, c.MakeNumeric(0.5))
	return anyvec.Sum(resVec)
}

func (t *TRPO) acceptable(r *anyrl.RolloutSet, grad anydiff.Grad,
	outs lazyseq.Tape) bool {
	c := creatorFromGrad(grad)
	inSeq := lazyseq.TapeRereader(c, r.Inputs)
	rewardSeq := lazyseq.TapeRereader(c, t.actionJudger().JudgeActions(r))
	oldOutSeq := lazyseq.TapeRereader(c, outs)
	newOutSeq := t.apply(inSeq, t.steppedPolicy(grad))
	sampledOut := lazyseq.TapeRereader(c, r.SampledOuts)

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
	}, rewardSeq, oldOutSeq, newOutSeq, sampledOut)

	outStats := lazyseq.Mean(mappedOut).Output()
	improvement := anyvec.Sum(outStats.Slice(0, 1))
	kl := anyvec.Sum(outStats.Slice(1, 2))

	targetImprovement := c.MakeNumeric(0)
	targetKL := c.MakeNumeric(t.targetKL())
	ops := c.NumOps()
	return ops.Greater(improvement, targetImprovement) &&
		ops.Less(kl, targetKL)
}

// storePolicyOutputs evaluates the policy on the inputs
// and stores the results to a tape.
// It also returns a lazyseq.Reuser for the output.
// The lazyseq.Reuser will be used, so you must call
// Reuse() on it before using it again.
func (n *NaturalPG) storePolicyOutputs(c anyvec.Creator,
	r *anyrl.RolloutSet) (lazyseq.Tape, lazyseq.Reuser) {
	tape, writer := lazyseq.ReferenceTape()

	out := lazyseq.MakeReuser(n.apply(lazyseq.TapeRereader(c, r.Inputs), n.Policy))
	for outVec := range out.Forward() {
		writer <- &anyseq.Batch{
			Present: outVec.Present,
			Packed:  outVec.Packed.Copy(),
		}
	}

	close(writer)
	return tape, out
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

func (t *TRPO) actionJudger() ActionJudger {
	if t.ActionJudger == nil {
		return &TotalJudger{Normalize: true}
	} else {
		return t.ActionJudger
	}
}
