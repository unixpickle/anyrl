package anyrl

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/lazyrnn"
)

func TestTRPO(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	r := rolloutsForTest(c)

	block := &anyrnn.LayerBlock{
		Layer: anynet.Net{
			anynet.NewFC(c, 3, 2),
			anynet.Tanh,
			anynet.NewFC(c, 2, 2),
		},
	}

	trpo := &TRPO{
		NaturalPG: NaturalPG{
			Policy:      block,
			Params:      block.Parameters(),
			ActionSpace: Softmax{},
			Iters:       14,
		},
	}
	grad := trpo.Run(r)

	policyGrad := anydiff.NewGrad(block.Parameters()...)
	PolicyGradient(trpo.ActionSpace, r, grad, func(in lazyrnn.Rereader) lazyrnn.Rereader {
		return lazyrnn.Lazify(anyrnn.Map(lazyrnn.Unlazify(in), block))
	})

	if dotGrad(grad, policyGrad).(float64) < 0 {
		t.Errorf("TRPO gave a direction of decrease")
	}
}
