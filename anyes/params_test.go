package anyes

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestAnynetParamsSerialize(t *testing.T) {
	cr := anyvec32.DefaultCreator{}
	params := AnynetParams{
		Params: []*anydiff.Var{
			{Vector: cr.MakeVectorData([]float32{1, 2, 3.5, -7})},
			{Vector: cr.MakeVectorData([]float32{2, 1})},
		},
	}
	data, err := params.Data()
	if err != nil {
		t.Error(err)
	}

	newParams := AnynetParams{
		Params: []*anydiff.Var{
			{Vector: cr.MakeVectorData([]float32{0, 0, 0, 0})},
			{Vector: cr.MakeVectorData([]float32{0, 0})},
		},
	}
	if err := newParams.SetData(data); err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(params, newParams) {
		t.Error("mismatching parameters")
	}
}

func TestAnynetParamsUpdate(t *testing.T) {
	cr := anyvec32.DefaultCreator{}
	params := AnynetParams{
		Params: []*anydiff.Var{
			{Vector: cr.MakeVectorData([]float32{1, 2, 3.5, -7})},
			{Vector: cr.MakeVectorData([]float32{2, 1})},
		},
	}
	params.Update([]float64{1, 2, 0.5, 1, 2, 3})

	expected := AnynetParams{
		Params: []*anydiff.Var{
			{Vector: cr.MakeVectorData([]float32{2, 4, 4, -6})},
			{Vector: cr.MakeVectorData([]float32{4, 4})},
		},
	}

	if !reflect.DeepEqual(params, expected) {
		t.Error("mismatching parameters")
	}
}
