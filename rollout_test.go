package anyrl

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/lazyseq"
)

func TestRolloutRemainingRewards(t *testing.T) {
	c := anyvec64.DefaultCreator{}

	tapeIn, writer := lazyseq.ReferenceTape()
	writer <- &anyseq.Batch{
		Present: []bool{true, false, true},
		Packed:  c.MakeVectorData([]float64{1, 0.5}),
	}
	writer <- &anyseq.Batch{
		Present: []bool{true, false, true},
		Packed:  c.MakeVectorData([]float64{0.5, -1}),
	}
	writer <- &anyseq.Batch{
		Present: []bool{true, false, false},
		Packed:  c.MakeVectorData([]float64{2}),
	}
	close(writer)

	tapeOut := (&RolloutSet{Rewards: tapeIn}).RemainingRewards()
	expected := []*anyseq.Batch{
		{
			Present: []bool{true, false, true},
			Packed:  c.MakeVectorData([]float64{3.5, -0.5}),
		},
		{
			Present: []bool{true, false, true},
			Packed:  c.MakeVectorData([]float64{2.5, -1}),
		},
		{
			Present: []bool{true, false, false},
			Packed:  c.MakeVectorData([]float64{2}),
		},
	}
	actual := tapeOut.ReadTape(0, -1)
	for i, x := range expected {
		a := <-actual
		if !reflect.DeepEqual(x.Present, a.Present) {
			t.Errorf("bad present map (%d): got %v expected %v", i,
				a.Present, x.Present)
			continue
		}
		diff := x.Packed.Copy()
		diff.Sub(a.Packed)
		if anyvec.AbsMax(diff).(float64) > 1e-4 {
			t.Errorf("bad vector (%d): got %v expected %v", i,
				a.Packed.Data(), x.Packed.Data())
		}
	}
	if _, ok := <-actual; ok {
		t.Errorf("expected end of stream")
	}
}
