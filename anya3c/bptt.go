package anya3c

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
)

// bptt implements back-propagation through time.
type bptt struct {
	Rollout     *rollout
	Worker      *worker
	Discount    float64
	Regularizer anypg.Regularizer
	Logger      Logger
}

// Run performs back-propagation through time.
//
// Since BPTT may be bootstrapped, the worker may be used
// to run the critic on the next observation.
func (b *bptt) Run() (grad anydiff.Grad, criticMSE anyvec.Numeric) {
	c := b.Worker.EnvObs.Creator()
	ops := c.NumOps()

	grad = anydiff.NewGrad(b.Worker.Agent.Params...)
	criticMSE = c.MakeNumeric(0)

	if len(b.Worker.Agent.Params) == 0 {
		return
	}

	advantages := b.Rollout.Advantages(b.Worker, b.Discount)
	stateUpstream := make([]anyrnn.StateGrad, 3)
	for t := len(advantages) - 1; t >= 0; t-- {
		outReses := b.Rollout.Outs[t]
		advantage := advantages[t]
		criticMSE = ops.Add(criticMSE, ops.Pow(advantage, c.MakeNumeric(2)))

		criticUpstream := c.MakeVector(1)
		criticUpstream.AddScalar(ops.Mul(advantage, c.MakeNumeric(2)))
		actorUpstream := b.actorUpstream(outReses[1].Output(),
			b.Rollout.Sampled[t], advantage)

		var baseUpstream1, baseUpstream2 anyvec.Vector
		baseUpstream1, stateUpstream[1] = outReses[1].Propagate(actorUpstream,
			stateUpstream[1], grad)
		baseUpstream2, stateUpstream[2] = outReses[2].Propagate(criticUpstream,
			stateUpstream[2], grad)

		baseUpstream1.Add(baseUpstream2)

		_, stateUpstream[0] = outReses[0].Propagate(baseUpstream1,
			stateUpstream[0], grad)
	}

	if b.Rollout.Beginning {
		for i, block := range b.Worker.blocks() {
			block.PropagateStart(stateUpstream[i], grad)
		}
	}

	criticMSE = ops.Div(criticMSE, c.MakeNumeric(float64(len(advantages))))

	return
}

func (b *bptt) actorUpstream(params, sampled anyvec.Vector,
	advantage anyvec.Numeric) anyvec.Vector {
	c := params.Creator()
	paramVar := anydiff.NewVar(params)
	grad := anydiff.NewGrad(paramVar)

	upstream := c.MakeVector(1)
	upstream.AddScalar(advantage)
	logProb := b.Worker.Agent.ActionSpace.LogProb(paramVar, sampled, 1)
	logProb.Propagate(upstream, grad)

	if b.Regularizer != nil {
		penalty := b.Regularizer.Regularize(paramVar, 1)
		upstream.SetData(c.MakeNumericList([]float64{1}))
		penalty.Propagate(upstream, grad)
		if b.Logger != nil {
			b.Logger.LogRegularize(b.Worker.ID, anyvec.Sum(penalty.Output()))
		}
	}

	return grad[paramVar]
}
