package anypg

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
)

const DefaultPPOEpsilon = 0.2

// PPOTerms represents the current value of the surrogate
// PPO objective function in terms of the advantage,
// critic, and regularization terms.
// The sum of the three terms exactly represents the
// objective function.
type PPOTerms struct {
	MeanAdvantage      anyvec.Numeric
	MeanCritic         anyvec.Numeric
	MeanRegularization anyvec.Numeric
}

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

	// PoolBase, if true, indicates that the output of the
	// Base function should be pooled to prevent multiple
	// forward/backward Base evaluations.
	//
	// If this is true, then the entire output of Base is
	// stored in memory.
	PoolBase bool
}

// Advantage computes the GAE estimator for a batch.
//
// You should call this once per batch.
// You should not call it between training steps in the
// same batch, since the advantage estimator will change
// as the value function is trained.
func (p *PPO) Advantage(r *anyrl.RolloutSet) lazyseq.Tape {
	first := <-r.Inputs.ReadTape(0, 1)
	if first == nil {
		tape, w := lazyseq.ReferenceTape()
		close(w)
		return tape
	}

	criticOut := p.Critic(p.applyBase(first.Packed.Creator(), r))

	estimatedValues := make([][]float64, len(r.Rewards))
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
	for i, rewSeq := range r.Rewards {
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

// Run computes the gradient for a PPO step.
// It takes a batch of rollouts and the precomputed
// advantages for that batch.
//
// This may be called multiple times per batch to fully
// maximize the objective.
//
// If p.Params is empty, then an empty gradient and nil
// PPOTerms are returned.
func (p *PPO) Run(r *anyrl.RolloutSet, adv lazyseq.Tape) (anydiff.Grad, *PPOTerms) {
	grad := anydiff.NewGrad(p.Params...)
	if len(grad) == 0 {
		return grad, nil
	}
	c := creatorFromGrad(grad)
	targetValues := (&QJudger{Discount: p.Discount}).JudgeActions(r)

	objective := p.runActorCritic(c, r, func(actor, critic lazyseq.Rereader) anydiff.Res {
		obj := lazyseq.MapN(
			func(n int, v ...anydiff.Res) anydiff.Res {
				actor, critic := v[0], v[1]
				oldOuts, actions := v[2], v[3]
				advantage, targets := v[4], v[5]

				ratios := anydiff.Exp(
					anydiff.Sub(
						p.ActionSpace.LogProb(actor, actions.Output(), n),
						p.ActionSpace.LogProb(oldOuts, actions.Output(), n),
					),
				)
				advTerm := p.clippedObjective(ratios, advantage)

				criticCoeff := -1.0
				if p.CriticWeight != 0 {
					criticCoeff *= p.CriticWeight
				}
				criticTerm := anydiff.Scale(
					anydiff.Square(anydiff.Sub(critic, targets)),
					c.MakeNumeric(criticCoeff),
				)

				var regTerm anydiff.Res
				if p.Regularizer != nil {
					regTerm = p.Regularizer.Regularize(actor, n)
				} else {
					regTerm = anydiff.NewConst(c.MakeVector(n))
				}

				cm := anynet.ConcatMixer{}
				return cm.Mix(cm.Mix(advTerm, criticTerm, n), regTerm, n)
			},
			actor,
			critic,
			lazyseq.TapeRereader(c, r.AgentOuts),
			lazyseq.TapeRereader(c, r.Actions),
			lazyseq.TapeRereader(c, adv),
			lazyseq.TapeRereader(c, targetValues.Tape(c)),
		)
		return lazyseq.Mean(obj)
	})
	objective.Propagate(anyvec.Ones(c, 3), grad)

	terms := &PPOTerms{
		MeanAdvantage:      anyvec.Sum(objective.Output().Slice(0, 1)),
		MeanCritic:         anyvec.Sum(objective.Output().Slice(1, 2)),
		MeanRegularization: anyvec.Sum(objective.Output().Slice(2, 3)),
	}

	return grad, terms
}

// runActorCritic computes the outputs of the actor and
// the critic and passes them to f.
// It returns the result of f.
//
// This may perform pooling, hence the unusual callback
// setup.
func (p *PPO) runActorCritic(c anyvec.Creator, r *anyrl.RolloutSet,
	f func(actor, critic lazyseq.Rereader) anydiff.Res) anydiff.Res {
	if p.PoolBase {
		baseOut := p.applyBase(c, r)
		return lazyseq.PoolToVec(baseOut, func(baseOut anyseq.Seq) anydiff.Res {
			actorOut := p.Actor(lazyseq.Lazify(baseOut))
			criticOut := p.Critic(lazyseq.Lazify(baseOut))
			return f(actorOut, criticOut)
		})
	} else {
		actorOut := p.Actor(p.applyBase(c, r))
		criticOut := p.Critic(p.applyBase(c, r))
		return f(actorOut, criticOut)
	}
}

func (p *PPO) applyBase(c anyvec.Creator, r *anyrl.RolloutSet) lazyseq.Reuser {
	if p.Base == nil {
		return lazyseq.MakeReuser(lazyseq.TapeRereader(c, r.Inputs))
	} else {
		return lazyseq.MakeReuser(p.Base(lazyseq.TapeRereader(c, r.Inputs)))
	}
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
