package anypg

import (
	"testing"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/lazyseq"
)

func TestTRPOImprovement(t *testing.T) {
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
			ActionSpace: anyrl.Softmax{},
			Iters:       14,
		},
	}
	grad := trpo.Run(r)

	pg := &PG{
		Policy: func(in lazyseq.Rereader) lazyseq.Rereader {
			return lazyseq.Lazify(anyrnn.Map(lazyseq.Unlazify(in), block))
		},
		Params:      anynet.AllParameters(block),
		ActionSpace: trpo.ActionSpace,
	}
	policyGrad := pg.Run(r)

	if dotGrad(grad, policyGrad).(float64) < 0 {
		t.Errorf("TRPO gave a direction of decrease")
	}
}
