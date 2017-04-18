package anyrl

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
)

// PolicyGradient approximates the gradient of a policy
// using rollouts sampled from that policy.
//
// The policy function should take a sequence of policy
// inputs and produce input parameters for the action
// space.
//
// The computed gradient is added to the grad argument.
func PolicyGradient(a ActionSpace, r *RolloutSet, grad anydiff.Grad,
	policy func(in lazyseq.Rereader) lazyseq.Rereader) {
	if len(grad) == 0 {
		return
	}
	c := creatorFromGrad(grad)

	inRereader := lazyseq.TapeRereader(c, r.Inputs)
	policyOut := policy(inRereader)

	selectedOuts := lazyseq.TapeRereader(c, r.SampledOuts)
	rewards := lazyseq.TapeRereader(c, r.RemainingRewards())

	scores := lazyseq.MapN(func(n int, v ...anydiff.Res) anydiff.Res {
		actionParams := v[0]
		selected := v[1]
		rewards := v[2]
		logProb := a.LogProb(actionParams, selected.Output(), n)
		return anydiff.Mul(logProb, rewards)
	}, policyOut, selectedOuts, rewards)

	score := lazyseq.Mean(scores)
	one := c.MakeVector(1)
	one.AddScalar(c.MakeNumeric(1))
	score.Propagate(one, grad)
}

func creatorFromGrad(g anydiff.Grad) anyvec.Creator {
	for _, v := range g {
		return v.Creator()
	}
	panic("empty gradient")
}
