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
	"github.com/unixpickle/lazyseq"
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
	outSeq := npg.apply(lazyseq.TapeRereader(c, r.Inputs), block)
	mapped := lazyseq.MapN(func(num int, v ...anydiff.Res) anydiff.Res {
		return npg.ActionSpace.KL(v[0], v[1], num)
	}, lazyseq.TapeRereader(c, stored), outSeq)
	expectedOutput := anyvec.Sum(lazyseq.Mean(mapped).Output()).(float64)

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

	// We have to use the actual gradient to avoid the
	// nullspace of the Fisher matrix.
	inGrad := anydiff.NewGrad(block.Parameters()...)
	PolicyGradient(npg.ActionSpace, r, inGrad,
		func(in lazyseq.Rereader) lazyseq.Rereader {
			return lazyseq.Lazify(anyrnn.Map(lazyseq.Unlazify(in), block))
		})
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

func BenchmarkFisher(b *testing.B) {
	c := anyvec64.DefaultCreator{}
	r := rolloutsForTest(c)

	block := &anyrnn.LayerBlock{
		Layer: anynet.Net{
			anynet.NewFC(c, 3, 256),
			anynet.Tanh,
			anynet.NewFC(c, 256, 256),
			anynet.Tanh,
			anynet.NewFC(c, 256, 30),
		},
	}

	npg := &NaturalPG{
		Policy:      block,
		Params:      block.Parameters(),
		ActionSpace: Softmax{},
	}

	inGrad := anydiff.NewGrad(block.Parameters()...)
	for _, vec := range inGrad {
		anyvec.Rand(vec, anyvec.Normal, nil)
		vec.Scale(c.MakeNumeric(0.0001))
	}
	stored := npg.storePolicyOutputs(c, r)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		npg.applyFisher(r, inGrad, stored)
	}
}

func rolloutsForTest(c anyvec.Creator) *RolloutSet {
	inputs, inputWriter := lazyseq.ReferenceTape()
	rewards, rewardWriter := lazyseq.ReferenceTape()
	sampledOuts, sampledOutsWriter := lazyseq.ReferenceTape()
	for i := 0; i < 3; i++ {
		vec := c.MakeVector(6)
		anyvec.Rand(vec, anyvec.Normal, nil)
		rew := c.MakeVector(2)
		anyvec.Rand(rew, anyvec.Uniform, nil)
		inputWriter <- &anyseq.Batch{
			Present: []bool{true, false, true},
			Packed:  vec,
		}
		rewardWriter <- &anyseq.Batch{
			Present: []bool{true, false, true},
			Packed:  rew,
		}
		sampled := make([]float64, 4)
		sampled[i%2] = 1
		sampled[(i+1)%2+2] = 1
		sampledOutsWriter <- &anyseq.Batch{
			Present: []bool{true, false, true},
			Packed:  c.MakeVectorData(c.MakeNumericList(sampled)),
		}
	}
	for i := 0; i < 8; i++ {
		vec := c.MakeVector(3)
		anyvec.Rand(vec, anyvec.Normal, nil)
		rew := c.MakeVector(1)
		anyvec.Rand(rew, anyvec.Uniform, nil)
		inputWriter <- &anyseq.Batch{
			Present: []bool{false, false, true},
			Packed:  vec,
		}
		rewardWriter <- &anyseq.Batch{
			Present: []bool{false, false, true},
			Packed:  rew,
		}
		sampled := make([]float64, 2)
		sampled[i%2] = 1
		sampledOutsWriter <- &anyseq.Batch{
			Present: []bool{false, false, true},
			Packed:  c.MakeVectorData(c.MakeNumericList(sampled)),
		}
	}
	close(inputWriter)
	close(rewardWriter)
	close(sampledOutsWriter)

	rollouts := &RolloutSet{
		Inputs:      inputs,
		Rewards:     rewards,
		SampledOuts: sampledOuts,
	}

	// TODO: create rewards and outputs as well.

	return rollouts
}
