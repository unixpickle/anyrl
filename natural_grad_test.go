package anyrl

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/lazyrnn"
)

func TestFisherDeterministic(t *testing.T) {
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
	stored := npg.storePolicyOutputs(c, r)

	grad1 := npg.applyFisher(r, inGrad, stored)
	mag1 := dotGrad(grad1, grad1).(float64)
	for i := 0; i < 1000; i++ {
		grad2 := npg.applyFisher(r, inGrad, stored)
		mag2 := dotGrad(grad2, grad2).(float64)
		correlation := dotGrad(grad1, grad2).(float64) / math.Sqrt(mag1*mag2)
		if correlation < 1-1e-3 {
			t.Errorf("correlation is too low: %f", correlation)
		}
	}
}

func TestFisher(t *testing.T) {
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
		vec.Scale(c.MakeNumeric(0.0001))
	}
	stored := npg.storePolicyOutputs(c, r)

	applied := npg.applyFisher(r, inGrad, stored)
	actualOutput := 0.5 * dotGrad(inGrad, applied).(float64)

	inGrad.AddToVars()
	outSeq := npg.apply(lazyrnn.TapeRereader(c, r.Inputs), block)
	mapped := lazyrnn.MapN(func(num int, v ...anydiff.Res) anydiff.Res {
		return npg.ActionSpace.KL(v[0], v[1], num)
	}, lazyrnn.TapeRereader(c, stored), outSeq)
	expectedOutput := anyvec.Sum(lazyrnn.Mean(mapped).Output()).(float64)

	diff := (actualOutput - expectedOutput) / actualOutput
	if math.Abs(diff) > 1e-3 {
		t.Errorf("expected %v but got %v", expectedOutput, actualOutput)
	}
}

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
