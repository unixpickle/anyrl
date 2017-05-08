package anyrl

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestSoftmaxSample(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	in := c.MakeVectorData([]float64{
		0.0902265411093121, -1.1492330740032015, -0.7417678904738725,
		0.1571149104608501, -1.3123382994428667, 1.2192607242291933,
	})
	expected := in.Copy()
	anyvec.LogSoftmax(expected, 3)
	anyvec.Exp(expected)

	actual := c.MakeVector(6)
	const numSamples = 100000
	for i := 0; i < numSamples; i++ {
		actual.Add(Softmax{}.Sample(in, 2))
	}
	actual.Scale(c.MakeNumeric(1.0 / numSamples))

	assertSimilar(t, actual, expected)
}

func TestSoftmaxLogProb(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	in := c.MakeVectorData([]float64{
		0.0902265411093121, -1.1492330740032015, -0.7417678904738725,
		0.1571149104608501, -1.3123382994428667, 1.2192607242291933,
	})

	sampled := c.MakeVectorData([]float64{0, 1, 0, 0, 0, 1})
	expected := in.Copy()
	anyvec.LogSoftmax(expected, 3)
	expected.Mul(sampled)
	expected = anyvec.SumCols(expected, 2)
	actual := Softmax{}.LogProb(anydiff.NewConst(in), sampled, 2).Output()

	assertSimilar(t, actual, expected)
}

func TestSoftmaxKL(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	in1 := c.MakeVectorData([]float64{
		0.0902265411093121, -1.1492330740032015, -0.7417678904738725,
		0.1571149104608501, -1.3123382994428667, 1.2192607242291933,
	})
	in2 := c.MakeVectorData([]float64{
		0.9313885780497441, -1.9309617520360562, 1.2151486203602158,
		0.6612636577085984, 0.3235493283220768, -0.0906927932047284,
	})

	actual := Softmax{}.KL(anydiff.NewConst(in1), anydiff.NewConst(in2), 2).Output()
	expected := c.MakeVectorData([]float64{0.315157204359214, 0.574736163236784})

	assertSimilar(t, actual, expected)
}

func TestSoftmaxEntropy(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	in := c.MakeVectorData([]float64{
		0.0902265411093121, -1.1492330740032015, -0.7417678904738725,
		0.1571149104608501, -1.3123382994428667, 1.2192607242291933,
	})

	actual := Softmax{}.Entropy(anydiff.NewConst(in), 2).Output()
	expected := c.MakeVectorData([]float64{0.963070145433149, 0.753250756925369})

	assertSimilar(t, actual, expected)
}

func TestBernoulliSample(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	in := c.MakeVectorData([]float64{
		0.0902265411093121, -1.1492330740032015, -0.7417678904738725,
		0.1571149104608501, -1.3123382994428667, 1.2192607242291933,
	})

	t.Run("Binary", func(t *testing.T) {
		expected := in.Copy()
		anyvec.Sigmoid(expected)

		actual := c.MakeVector(6)
		const numSamples = 100000
		for i := 0; i < numSamples; i++ {
			actual.Add((&Bernoulli{}).Sample(in, 2))
		}
		actual.Scale(c.MakeNumeric(1.0 / numSamples))

		assertSimilar(t, actual, expected)
	})

	t.Run("OneHot", func(t *testing.T) {
		softmaxed := make([]float64, in.Len()*2)
		for i, x := range in.Data().([]float64) {
			softmaxed[i*2] = 1 - 1/(1+math.Exp(-x))
			softmaxed[i*2+1] = 1 / (1 + math.Exp(-x))
		}
		expected := c.MakeVectorData(softmaxed)

		actual := c.MakeVector(6)
		const numSamples = 100000
		for i := 0; i < numSamples; i++ {
			actual.Add((&Bernoulli{OneHot: true}).Sample(in, 2))
		}
		actual.Scale(c.MakeNumeric(1.0 / numSamples))

		assertSimilar(t, actual, expected)
	})
}

func TestBernoulliLogProb(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	in := c.MakeVectorData([]float64{
		0.0902265411093121, -1.1492330740032015, -0.7417678904738725,
		0.1571149104608501, -1.3123382994428667, 1.2192607242291933,
	})

	t.Run("Binary", func(t *testing.T) {
		sampledOuts := c.MakeVectorData([]float64{
			1, 0, 1,
			0, 1, 1,
		})
		expected := c.MakeVectorData([]float64{
			-2.05560356588594, -2.58436117597908,
		})
		actual := (&Bernoulli{}).LogProb(anydiff.NewConst(in), sampledOuts, 2)
		assertSimilar(t, actual.Output(), expected)
	})

	t.Run("OneHot", func(t *testing.T) {
		sampledOuts := c.MakeVectorData([]float64{
			0, 1, 1, 0, 0, 1,
			1, 0, 0, 1, 0, 1,
		})
		expected := c.MakeVectorData([]float64{
			-2.05560356588594, -2.58436117597908,
		})
		actual := (&Bernoulli{OneHot: true}).LogProb(anydiff.NewConst(in),
			sampledOuts, 2)
		assertSimilar(t, actual.Output(), expected)
	})
}

func TestBernoulliKL(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	in1 := c.MakeVectorData([]float64{
		0.0902265411093121, -1.1492330740032015, -0.7417678904738725,
		0.1571149104608501, -1.3123382994428667, 1.2192607242291933,
	})
	in2 := c.MakeVectorData([]float64{
		0.9313885780497441, -1.9309617520360562, 1.2151486203602158,
		0.6612636577085984, 0.3235493283220768, -0.0906927932047284,
	})

	softmaxIns := make([]anydiff.Res, 2)
	for i, in := range []anyvec.Vector{in1, in2} {
		si := make([]float64, in.Len()*2)
		for j, x := range in.Data().([]float64) {
			si[2*j] = -x
		}
		softmaxIns[i] = anydiff.NewConst(c.MakeVectorData(si))
	}

	actual := (&Bernoulli{}).KL(anydiff.NewConst(in1),
		anydiff.NewConst(in2), 2).Output()
	expected := anyvec.SumCols(
		Softmax{}.KL(softmaxIns[0], softmaxIns[1], 6).Output(),
		2,
	)

	assertSimilar(t, actual, expected)
}

func TestBernoulliEntropy(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	in := c.MakeVectorData([]float64{
		0.0902265411093121, -1.1492330740032015, -0.7417678904738725,
		0.1571149104608501, -1.3123382994428667, 1.2192607242291933,
	})

	si := make([]float64, in.Len()*2)
	for j, x := range in.Data().([]float64) {
		si[2*j] = -x
	}
	softmaxIns := anydiff.NewConst(c.MakeVectorData(si))

	actual := (&Bernoulli{}).Entropy(anydiff.NewConst(in), 2).Output()
	expected := anyvec.SumCols(
		Softmax{}.Entropy(softmaxIns, 6).Output(),
		2,
	)

	assertSimilar(t, actual, expected)
}

func TestTupleSample(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	in := c.MakeVectorData([]float64{
		2.079628004581360, -0.882835230320012, -0.451191417094636, 0.958881016523292,
		-0.972656198661668, -1.480224804168850, -0.345374978440303, -1.702737439027942,
		1.438766876689676, 0.414036994920725, -0.833089098102292, -0.411917530033889,
	})
	expected := c.MakeVectorData([]float64{
		0.8839503917982873, 0.0456926616438972, 0.0703569465578157, 0.2771022899363433, 0.7228977100636567,
		0.2878136726853974, 0.1732515691914487, 0.5389347581231539, 0.8458919221061184, 0.1541080778938816,
		0.6839879739834515, 0.2454787667809506, 0.0705332592355978, 0.6015475773307006, 0.3984524226692994,
	})
	space := &Tuple{
		Spaces:      []interface{}{Softmax{}, &Bernoulli{OneHot: true}},
		ParamSizes:  []int{3, 1},
		SampleSizes: []int{3, 2},
	}

	actual := c.MakeVector(15)
	const numSamples = 100000
	for i := 0; i < numSamples; i++ {
		actual.Add(space.Sample(in, 3))
	}
	actual.Scale(c.MakeNumeric(1.0 / numSamples))

	assertSimilar(t, actual, expected)
}

func TestTupleLogProb(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	in := c.MakeVectorData([]float64{
		2.079628004581360, -0.882835230320012, -0.451191417094636, 0.958881016523292,
		-0.972656198661668, -1.480224804168850, -0.345374978440303, -1.702737439027942,
		1.438766876689676, 0.414036994920725, -0.833089098102292, -0.411917530033889,
	})
	sampled := c.MakeVectorData([]float64{
		0, 1, 0, 0, 1,
		1, 0, 0, 1, 0,
		0, 0, 1, 1, 0,
	})
	expected := c.MakeVectorData([]float64{
		-3.41030511738499, -1.41280565724833, -3.15992056709048,
	})
	space := &Tuple{
		Spaces:      []interface{}{Softmax{}, &Bernoulli{OneHot: true}},
		ParamSizes:  []int{3, 1},
		SampleSizes: []int{3, 2},
	}

	actual := space.LogProb(anydiff.NewConst(in), sampled, 3)

	assertSimilar(t, actual.Output(), expected)
}

func TestTupleKL(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	params1 := c.MakeVectorData([]float64{
		2.079628004581360, -0.882835230320012, -0.451191417094636, 0.958881016523292,
		-0.972656198661668, -1.480224804168850, -0.345374978440303, -1.702737439027942,
		1.438766876689676, 0.414036994920725, -0.833089098102292, -0.411917530033889,
	})
	params2 := c.MakeVectorData([]float64{
		2, 1, 0.4, -1,
		-1, 1.5, -0.3, 0,
		1, 1, -2, -0.4,
	})
	space := &Tuple{
		Spaces:      []interface{}{Softmax{}, &Bernoulli{OneHot: true}},
		ParamSizes:  []int{3, 1},
		SampleSizes: []int{3, 2},
	}

	approxKL := c.MakeVector(3)

	const numSamples = 100000
	for i := 0; i < numSamples; i++ {
		sample := space.Sample(params1, 3)
		prob1 := space.LogProb(anydiff.NewConst(params1), sample, 3).Output()
		prob2 := space.LogProb(anydiff.NewConst(params2), sample, 3).Output()
		prob1.Sub(prob2)
		approxKL.Add(prob1)
	}
	approxKL.Scale(c.MakeNumeric(1.0 / numSamples))

	actualKL := space.KL(anydiff.NewConst(params1), anydiff.NewConst(params2), 3)

	assertSimilar(t, actualKL.Output(), approxKL)
}

func TestTupleEntropy(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	params := c.MakeVectorData([]float64{
		2.079628004581360, -0.882835230320012, -0.451191417094636, 0.958881016523292,
		-0.972656198661668, -1.480224804168850, -0.345374978440303, -1.702737439027942,
		1.438766876689676, 0.414036994920725, -0.833089098102292, -0.411917530033889,
	})
	space := &Tuple{
		Spaces:      []interface{}{Softmax{}, &Bernoulli{OneHot: true}},
		ParamSizes:  []int{3, 1},
		SampleSizes: []int{3, 2},
	}

	approxEnt := c.MakeVector(3)

	const numSamples = 100000
	for i := 0; i < numSamples; i++ {
		sample := space.Sample(params, 3)
		prob := space.LogProb(anydiff.NewConst(params), sample, 3).Output()
		approxEnt.Sub(prob)
	}
	approxEnt.Scale(c.MakeNumeric(1.0 / numSamples))

	actualEnt := space.Entropy(anydiff.NewConst(params), 3)

	assertSimilar(t, actualEnt.Output(), approxEnt)
}

func assertSimilar(t *testing.T, actual, expected anyvec.Vector) {
	diff := actual.Copy()
	diff.Sub(expected)
	if anyvec.AbsMax(diff).(float64) > 1e-2 {
		t.Errorf("expected %v but got %v", expected.Data(), actual.Data())
	}
}
