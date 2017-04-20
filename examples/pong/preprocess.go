package main

import (
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
)

const (
	FrameWidth  = 160
	FrameHeight = 210

	PreprocessedSize = 80 * 80
)

type PreprocessEnv struct {
	Env        anyrl.Env
	Subsampler anyvec.Mapper
	Last       anyvec.Vector
}

func (p *PreprocessEnv) Reset() (observation anyvec.Vector, err error) {
	p.Last, err = p.Env.Reset()
	p.Last = p.simplifyImage(p.Last)
	return p.Last, err
}

func (p *PreprocessEnv) Step(action anyvec.Vector) (observation anyvec.Vector,
	reward float64, done bool, err error) {
	observation, reward, done, err = p.Env.Step(action)
	if observation != nil {
		observation = p.simplifyImage(observation)
		backup := observation.Copy()
		observation.Sub(p.Last)
		p.Last = backup
	}
	return
}

func (p *PreprocessEnv) simplifyImage(in anyvec.Vector) anyvec.Vector {
	if p.Subsampler == nil {
		p.Subsampler = makeInputSubsampler(in.Creator())
	}
	return preprocessImage(p.Subsampler, in)
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
