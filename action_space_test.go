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

func assertSimilar(t *testing.T, actual, expected anyvec.Vector) {
	diff := actual.Copy()
	diff.Sub(expected)
	if anyvec.AbsMax(diff).(float64) > 1e-2 {
		t.Errorf("expected %v but got %v", expected.Data(), actual.Data())
	}
}
