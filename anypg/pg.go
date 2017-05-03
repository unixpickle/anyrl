package anypg

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
)

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

func creatorFromGrad(g anydiff.Grad) anyvec.Creator {
	for _, v := range g {
		return v.Creator()
	}
	panic("empty gradient")
}
