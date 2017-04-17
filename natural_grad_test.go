package anyrl

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/lazyrnn"
)

func TestConjugateGradients(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	r := rolloutsForTest(c)

	block := &anyrnn.LayerBlock{
		Layer: anynet.Net{
			anynet.NewFC(c, 3, 2),
			anynet.Tanh,
			anynet.NewFC(c, 2, 2),
		},
	}

	npg := &NaturalPG{
		Policy:      block,
		Params:      block.Parameters(),
		ActionSpace: Softmax{},
		Iters:       14,
	}

	inGrad := anydiff.NewGrad(block.Parameters()...)
	for _, vec := range inGrad {
		anyvec.Rand(vec, anyvec.Normal, nil)
	}
	solvedGrad := copyGrad(inGrad)

	npg.conjugateGradients(r, solvedGrad)

	// Check that F*solvedGrad = inGrad.
	actualProduct := npg.applyFisher(r, solvedGrad, npg.storePolicyOutputs(c, r))
	expectedProduct := inGrad

	if len(actualProduct) != len(expectedProduct) {
		t.Fatalf("should have %d vars but have %d", len(expectedProduct),
			len(actualProduct))
	}

	for variable, actual := range actualProduct {
		expected := expectedProduct[variable]
		diff := actual.Copy()
		diff.Sub(expected)
		if anyvec.AbsMax(diff).(float64) > 1e-3 {
			t.Errorf("variable grad should be %v but it's %v", expected.Data(),
				actual.Data())
		}
	}
}

func rolloutsForTest(c anyvec.Creator) *RolloutSet {
	inputs, inputWriter := lazyrnn.ReferenceTape()
	inputWriter <- &anyseq.Batch{
		Present: []bool{true, false, true},
		Packed:  c.MakeVectorData(c.MakeNumericList([]float64{1, 2, -3, 2, -1, 0})),
	}
	close(inputWriter)

	rollouts := &RolloutSet{Inputs: inputs}

	// TODO: create rewards and outputs as well.

	return rollouts
}
