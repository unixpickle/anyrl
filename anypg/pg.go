package anypg

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyrl"
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

	// Regularizer is used to regularize the action space.
	//
	// If nil, no regularization is used.
	Regularizer Regularizer
}

// Run performs policy gradients on the rollouts.
func (p *PG) Run(r *anyrl.RolloutSet) anydiff.Grad {
	grad := anydiff.NewGrad(p.Params...)
	if len(grad) == 0 {
		return grad
	}
	c := r.Creator()

	policyOut := p.Policy(lazyseq.TapeRereader(r.Inputs))

	selectedOuts := lazyseq.TapeRereader(r.Actions)
	rewards := lazyseq.TapeRereader(p.actionJudger().JudgeActions(r).Tape(c))

	scores := lazyseq.MapN(func(n int, v ...anydiff.Res) anydiff.Res {
		actionParams := v[0]
		selected := v[1]
		rewards := v[2]
		logProb := p.ActionSpace.LogProb(actionParams, selected.Output(), n)
		cost := anydiff.Mul(logProb, rewards)
		if p.Regularizer != nil {
			cost = anydiff.Add(cost, p.Regularizer.Regularize(actionParams, n))
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
