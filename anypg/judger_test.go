package anypg

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/lazyseq"
)

func TestQJudger(t *testing.T) {
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

	tapeOut := (&QJudger{}).JudgeActions(&anyrl.RolloutSet{Rewards: tapeIn})
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
	testTapeEquiv(t, tapeOut, expected)
}

func TestQJudgerDiscounted(t *testing.T) {
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

	j := &QJudger{Discount: 0.5}
	tapeOut := j.JudgeActions(&anyrl.RolloutSet{Rewards: tapeIn})
	expected := []*anyseq.Batch{
		{
			Present: []bool{true, false, true},
			Packed:  c.MakeVectorData([]float64{1.75, 0}),
		},
		{
			Present: []bool{true, false, true},
			Packed:  c.MakeVectorData([]float64{1.5, -1}),
		},
		{
			Present: []bool{true, false, false},
			Packed:  c.MakeVectorData([]float64{2}),
		},
	}
	testTapeEquiv(t, tapeOut, expected)
}

func TestTotalJudger(t *testing.T) {
	c := anyvec64.DefaultCreator{}

	tapeIn, writer := lazyseq.ReferenceTape()
	writer <- &anyseq.Batch{
		Present: []bool{true, false, true, true},
		Packed:  c.MakeVectorData([]float64{1, 2, -1}),
	}
	writer <- &anyseq.Batch{
		Present: []bool{true, false, true, true},
		Packed:  c.MakeVectorData([]float64{2, -1, -1}),
	}
	writer <- &anyseq.Batch{
		Present: []bool{true, false, false, true},
		Packed:  c.MakeVectorData([]float64{3, -2}),
	}
	writer <- &anyseq.Batch{
		Present: []bool{true, false, false, false},
		Packed:  c.MakeVectorData([]float64{1}),
	}
	close(writer)

	// Sums: 7 1 -4; mean=1.33333; std=4.4969
	// Normalized:  1.260131  -0.074125  -1.186005

	judger := &TotalJudger{Normalize: true}
	tapeOut := judger.JudgeActions(&anyrl.RolloutSet{Rewards: tapeIn})
	expected := []*anyseq.Batch{
		{
			Present: []bool{true, false, true, true},
			Packed:  c.MakeVectorData([]float64{1.260131, -0.074125, -1.186005}),
		},
		{
			Present: []bool{true, false, true, true},
			Packed:  c.MakeVectorData([]float64{1.260131, -0.074125, -1.186005}),
		},
		{
			Present: []bool{true, false, false, true},
			Packed:  c.MakeVectorData([]float64{1.260131, -1.186005}),
		},
		{
			Present: []bool{true, false, false, false},
			Packed:  c.MakeVectorData([]float64{1.260131}),
		},
	}
	testTapeEquiv(t, tapeOut, expected)
}

func testTapeEquiv(t *testing.T, tapeOut lazyseq.Tape, expected []*anyseq.Batch) {
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
