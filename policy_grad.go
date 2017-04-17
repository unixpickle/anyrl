package anyrl

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyrnn"
)

// PolicyGradient approximates the gradient of a policy
// using rollouts sampled from that policy.
//
// The policy function should take a sequence of policy
// inputs and produce input parameters for the action
// space.
//
// The computed gradient is added to the grad argument.
func PolicyGradient(c anyvec.Creator, a ActionSpace, r *RolloutSet,
	grad anydiff.Grad, policy func(in lazyrnn.Rereader) lazyrnn.Seq) {
	inRereader := lazyrnn.TapeRereader(c, r.Inputs)
	policyOut := policy(inRereader)

	policyRereader := lazyrnn.Lazify(lazyrnn.Unlazify(policyOut))
	selectedOuts := lazyrnn.TapeRereader(c, r.SampledOuts)
	rewards := lazyrnn.TapeRereader(c, r.RemainingRewards())

	scores := lazyrnn.MapN(func(n int, v ...anydiff.Res) anydiff.Res {
		actionParams := v[0]
		selected := v[1]
		rewards := v[2]
		logProb := a.LogProb(actionParams, selected.Output(), n)
		return anydiff.Mul(logProb, rewards)
	}, policyRereader, selectedOuts, rewards)

	score := lazyrnn.Mean(scores)
	one := c.MakeVector(1)
	one.AddScalar(c.MakeNumeric(1))
	score.Propagate(one, grad)
}
