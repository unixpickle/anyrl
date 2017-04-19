package main

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/serializer"
)

const (
	FrameWidth  = 160
	FrameHeight = 210

	PreprocessedSize = 80 * 80
)

func init() {
	serializer.RegisterTypedDeserializer(PreprocessLayer{}.SerializerType(),
		DeserializePreprocessLayer)
}

type PreprocessLayer struct{}

func DeserializePreprocessLayer(d []byte) (PreprocessLayer, error) {
	return PreprocessLayer{}, nil
}

func (p PreprocessLayer) Apply(in anydiff.Res, n int) anydiff.Res {
	cr := in.Output().Creator()
	subsampler := makeInputSubsampler(cr)
	imgSize := FrameWidth * FrameHeight * 3
	var subsampled []anyvec.Vector
	for i := 0; i < n; i++ {
		inImg := in.Output().Slice(i*imgSize, (i+1)*imgSize)
		outImg := preprocessImage(subsampler, inImg)
		subsampled = append(subsampled, outImg)
	}
	return anydiff.NewConst(cr.Concat(subsampled...))
}

func (p PreprocessLayer) SerializerType() string {
	return "github.com/unixpickle/anyrl/examples/pong.preprocessLayer"
}

func (p PreprocessLayer) Serialize() ([]byte, error) {
	return []byte{}, nil
}

func preprocessImage(sampler anyvec.Mapper, image anyvec.Vector) anyvec.Vector {
	// Logic taken from:
	// https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5#file-pg-pong-py-L32.

	cr := image.Creator()

	// Crop image to 80x80
	out := cr.MakeVector(sampler.OutSize())
	sampler.Map(image, out)

	// Find background pixels
	bg1 := out.Copy()
	anyvec.EqualTo(out, cr.MakeNumeric(144))
	anyvec.EqualTo(bg1, cr.MakeNumeric(109))
	out.Add(bg1)

	// Set everything that's not a bg to 1.
	anyvec.Complement(out)

	return out
}

func makeInputSubsampler(cr anyvec.Creator) anyvec.Mapper {
	// Creating a mapping which does the following:
	//  - crop box from x=35 to x=195
	//  - subsample by factor of 2
	//  - select the red channel
	mapping := make([]int, 0, 80*80)
	for y := 35; y < 195; y += 2 {
		for x := 0; x < FrameWidth; x += 2 {
			sourceIdx := y*FrameWidth*3 + x*3
			mapping = append(mapping, sourceIdx)
		}
	}
	return cr.MakeMapper(FrameWidth*FrameHeight*3, mapping)
}
