package anypg

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
)

const DefaultPPOEpsilon = 0.2

// PPO implements Proximal Policy Optimization.
// See https://arxiv.org/abs/1707.06347.
type PPO struct {
	// Params specifies which parameters to include in
	// the gradients.
	Params []*anydiff.Var

	// Base is the part of the agent shared by both the
	// actor and the critic.
	// Its outputs are fed into Actor and Critic.
	//
	// If nil, the identity mapping is used.
	Base func(obses lazyseq.Rereader) lazyseq.Rereader

	// Actor is the policy part of the agent.
	Actor func(baseOut lazyseq.Rereader) lazyseq.Rereader

	// Critic estimates the value function.
	Critic func(baseOut lazyseq.Rereader) lazyseq.Rereader

	// ActionSpace determines log-likelihoods of actions.
	ActionSpace anyrl.LogProber

	// CriticWeight is the importance assigned to the
	// critic's loss during training.
	//
	// If 0, a default of 1 is used.
	CriticWeight float64

	// Regularizer can be used to encourage exploration.
	Regularizer Regularizer

	// Discount is the reward discount factor.
	Discount float64

	// Lambda is the GAE coefficient.
	Lambda float64

	// Epsilon is the amount by which the probability
	// ratio should change.
	//
	// If 0, DefaultPPOEpsilon is used.
	Epsilon float64
}

// Run computes the gradient for a PPO step.
//
// This may be called multiple times per batch to fully
// maximize the objective.
func (p *PPO) Run(r *anyrl.RolloutSet) anydiff.Grad {
	grad := anydiff.NewGrad(p.Params...)
	if len(grad) == 0 {
		return grad
	}
	c := creatorFromGrad(grad)

	var baseOut lazyseq.Reuser
	if p.Base == nil {
		baseOut = lazyseq.MakeReuser(lazyseq.TapeRereader(c, r.Inputs))
	} else {
		baseOut = lazyseq.MakeReuser(p.Base(lazyseq.TapeRereader(c, r.Inputs)))
	}

	criticOut := lazyseq.MakeReuser(p.Critic(baseOut))
	advantage := p.advantage(criticOut, r.Rewards)
	criticOut.Reuse()
	baseOut.Reuse()
	actorOut := p.Actor(baseOut)

	targetValues := (&QJudger{Discount: p.Discount}).JudgeActions(r)

	obj := lazyseq.MapN(
		func(n int, v ...anydiff.Res) anydiff.Res {
			actor, critic := v[0], v[1]
			oldOuts, actions := v[2], v[3]
			advantage, targets := v[4], v[5]

			criticCoeff := -1.0
			if p.CriticWeight != 0 {
				criticCoeff *= p.CriticWeight
			}
			criticCost := anydiff.Scale(
				anydiff.Square(anydiff.Sub(critic, targets)),
				c.MakeNumeric(criticCoeff),
			)

			rat := anydiff.Exp(
				anydiff.Sub(
					p.ActionSpace.LogProb(actor, actions.Output(), n),
					p.ActionSpace.LogProb(oldOuts, actions.Output(), n),
				),
			)

			obj := anydiff.Add(p.clippedObjective(rat, advantage), criticCost)
			if p.Regularizer != nil {
				regTerm := p.Regularizer.Regularize(actor, n)
				obj = anydiff.Add(obj, regTerm)
			}

			return obj
		},
		actorOut,
		criticOut,
		lazyseq.TapeRereader(c, r.AgentOuts),
		lazyseq.TapeRereader(c, r.Actions),
		lazyseq.TapeRereader(c, advantage),
		lazyseq.TapeRereader(c, targetValues.Tape(c)),
	)

	objective := lazyseq.Mean(obj)
	objective.Propagate(anyvec.Ones(c, 1), grad)
	return grad
}

func (p *PPO) advantage(criticOut lazyseq.Seq, r anyrl.Rewards) lazyseq.Tape {
	estimatedValues := make([][]float64, len(r))
	for outBatch := range criticOut.Forward() {
		comps := vectorToComponents(outBatch.Packed)
		for i, pres := range outBatch.Present {
			if pres {
				estimatedValues[i] = append(estimatedValues[i], comps[0])
				comps = comps[1:]
			}
		}
	}

	var res [][]float64
	for i, rewSeq := range r {
		valSeq := estimatedValues[i]
		advantages := make([]float64, len(rewSeq))
		var accumulation float64
		for t := len(rewSeq) - 1; t >= 0; t-- {
			delta := rewSeq[t] - valSeq[t]
			if t+1 < len(rewSeq) {
				delta += p.Discount * valSeq[t+1]
			}
			accumulation *= p.Discount * p.Lambda
			accumulation += delta
			advantages[t] = accumulation
		}
		res = append(res, advantages)
	}

	return anyrl.Rewards(res).Tape(criticOut.Creator())
}

func (p *PPO) clippedObjective(ratios, advantages anydiff.Res) anydiff.Res {
	epsilon := p.Epsilon
	if epsilon == 0 {
		epsilon = DefaultPPOEpsilon
	}
	c := ratios.Output().Creator()
	return anydiff.Pool(ratios, func(ratios anydiff.Res) anydiff.Res {
		clipped := anydiff.ClipRange(ratios, c.MakeNumeric(1-epsilon),
			c.MakeNumeric(1+epsilon))
		return anydiff.ElemMin(
			anydiff.Mul(clipped, advantages),
			anydiff.Mul(ratios, advantages),
		)
	})
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
