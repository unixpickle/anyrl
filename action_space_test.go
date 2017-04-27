package anyrl

import (
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

func assertSimilar(t *testing.T, actual, expected anyvec.Vector) {
	diff := actual.Copy()
	diff.Sub(expected)
	if anyvec.AbsMax(diff).(float64) > 1e-2 {
		t.Errorf("expected %v but got %v", expected.Data(), actual.Data())
	}
}
