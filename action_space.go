package anyrl

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

// A Sampler samples from a parametric distribution.
//
// For an example, see Softmax.
type Sampler interface {
	// Sample samples a batch of vectors given a batch
	// of parameter vectors.
	Sample(params anyvec.Vector, batchSize int) anyvec.Vector
}

// A LogProber can compute the log-likelihood of a given
// output of a parametric distribution.
//
// For an example, see Softmax.
type LogProber interface {
	// LogProb produces, for each parameter-output pair
	// in the batch, a log-probability of the parameters
	// producing that output.
	//
	// The natural logarithm should be used.
	//
	// For continuous distributions, this is the log of
	// the density rather than of the probability.
	LogProb(params anydiff.Res, output anyvec.Vector,
		batchSize int) anydiff.Res
}

// A KLer can compute the KL divergence between
// parameteric probability distributions given the
// parameters of both distributions.
//
// For an example, see Softmax.
type KLer interface {
	// KL computes the KL divergence between action space
	// distributions, given the parameters for each.
	//
	// This is batched, just like LogProber.LogProb.
	// It produces one value per entry in the batch.
	KL(params1, params2 anydiff.Res, batchSize int) anydiff.Res
}

// An Entropyer can compute the entropy of a parametric
// probability distribution.
type Entropyer interface {
	// Entropy computes the entropy for each parameter
	// vector in a batch.
	//
	// Entropy should be measured in nats.
	Entropy(params anydiff.Res, batchSize int) anydiff.Res
}

// Softmax is an action space which applies the softmax
// function to obtain a categorical distribution.
// It produces one-hot vector samples.
type Softmax struct{}

// Sample samples one-hot vectors from the softmax
// distribution.
func (s Softmax) Sample(params anyvec.Vector, batch int) anyvec.Vector {
	if params.Len()%batch != 0 {
		panic("batch size must divide parameter count")
	}

	chunkSize := params.Len() / batch
	p := params.Copy()
	anyvec.LogSoftmax(p, chunkSize)
	anyvec.Exp(p)

	var oneHots []anyvec.Vector
	for i := 0; i < batch; i++ {
		subset := p.Slice(i*chunkSize, (i+1)*chunkSize)
		oneHots = append(oneHots, sampleProbabilities(subset))
	}

	return p.Creator().Concat(oneHots...)
}

// LogProb computes the output log probabilities.
func (s Softmax) LogProb(params anydiff.Res, output anyvec.Vector,
	batchSize int) anydiff.Res {
	if params.Output().Len() != output.Len() {
		panic("length mismatch")
	}
	if params.Output().Len()%batchSize != 0 {
		panic("batch size does not divide param count")
	}
	chunkSize := params.Output().Len() / batchSize
	logs := anydiff.LogSoftmax(params, chunkSize)
	return batchedDot(logs, anydiff.NewConst(output), batchSize)
}

// KL computes the KL divergences between two batches of
// softmax distributions.
func (s Softmax) KL(params1, params2 anydiff.Res, batchSize int) anydiff.Res {
	if params1.Output().Len() != params2.Output().Len() {
		panic("length mismatch")
	}
	if params1.Output().Len()%batchSize != 0 {
		panic("batch size does not divide param count")
	}
	chunkSize := params1.Output().Len() / batchSize
	log1 := anydiff.LogSoftmax(params1, chunkSize)
	log2 := anydiff.LogSoftmax(params2, chunkSize)
	return anydiff.Pool(log1, func(log1 anydiff.Res) anydiff.Res {
		probs := anydiff.Exp(log1)
		diff := anydiff.Sub(log1, log2)
		return batchedDot(probs, diff, batchSize)
	})
}

// Entropy computes the entropy of the distributions.
func (s Softmax) Entropy(params anydiff.Res, batchSize int) anydiff.Res {
	chunkSize := params.Output().Len() / batchSize
	return anydiff.Pool(params, func(params anydiff.Res) anydiff.Res {
		logProbs := anydiff.LogSoftmax(params, chunkSize)
		probs := anydiff.Exp(logProbs)
		return anydiff.Scale(batchedDot(probs, logProbs, batchSize),
			params.Output().Creator().MakeNumeric(-1))
	})
}

// Bernoulli is an action space for binary actions or
// lists of binary actions.
// It can be used with a one-hot representation (similar
// to softmax) or with binary action spaces where each
// binary value is a single number.
//
// The Bernoulli distribution is implemented via the
// logistic sigmoid, 1/(1+exp(-x)).
// When an input x is fed in, the logistic sigmoid is used
// to compute the probability of a 1.
type Bernoulli struct {
	// OneHot, if true, indicates that samples should be
	// one-hot vectors with two components.
	// If false, samples are binary values (0 or 1).
	OneHot bool
}

// Sample samples Bernoulli random variables.
func (b *Bernoulli) Sample(params anyvec.Vector, batch int) anyvec.Vector {
	probs := params.Copy()
	anyvec.Sigmoid(probs)

	cutoffs := params.Creator().MakeVector(params.Len())
	anyvec.Rand(cutoffs, anyvec.Uniform, nil)

	// Turn probs into a sampled binary vector.
	probs.Sub(cutoffs)
	anyvec.GreaterThan(probs, params.Creator().MakeNumeric(0))

	if b.OneHot {
		return pairWithComplement(anydiff.NewConst(probs)).Output()
	} else {
		return probs
	}
}

// LogProb computes the output log probabilities.
func (b *Bernoulli) LogProb(params anydiff.Res, output anyvec.Vector,
	batch int) anydiff.Res {
	offOn := b.offOnProbs(params)
	var outputMask anydiff.Res = anydiff.NewConst(output)
	if !b.OneHot {
		outputMask = pairWithComplement(outputMask)
	}
	return batchedDot(offOn, outputMask, batch)
}

// KL computes the KL divergences between two batches of
// Bernoulli distributions.
func (b *Bernoulli) KL(params1, params2 anydiff.Res, batchSize int) anydiff.Res {
	offOn1 := b.offOnProbs(params1)
	offOn2 := b.offOnProbs(params2)
	return anydiff.Pool(offOn1, func(offOn1 anydiff.Res) anydiff.Res {
		diff := anydiff.Sub(offOn1, offOn2)
		probs1 := anydiff.Exp(offOn1)
		return batchedDot(probs1, diff, batchSize)
	})
}

// Entropy computes the information entropy of Bernoulli
// distributions.
func (b *Bernoulli) Entropy(params anydiff.Res, batchSize int) anydiff.Res {
	return anydiff.Pool(params, func(params anydiff.Res) anydiff.Res {
		logs := b.offOnProbs(params)
		probs := anydiff.Exp(logs)
		return anydiff.Scale(batchedDot(probs, logs, batchSize),
			params.Output().Creator().MakeNumeric(-1))
	})
}

func (b *Bernoulli) offOnProbs(params anydiff.Res) anydiff.Res {
	return anydiff.Pool(params, func(params anydiff.Res) anydiff.Res {
		c := params.Output().Creator()
		onProbs := anydiff.LogSigmoid(params)
		offProbs := anydiff.LogSigmoid(anydiff.Scale(params, c.MakeNumeric(-1)))
		return sideBySide(offProbs, onProbs)
	})
}

// Gaussian is a continuous action space with values
// sampled from a parametric Gaussian distribution.
//
// For each component of the output, there is a parameter
// for a mean and one for a variance (in that order).
// The parameters may be any values, even though variance
// can only be positive.
// To deal with this, the variance parameter is fed into
// the exponential function.
type Gaussian struct{}

// Sample samples continuous values from the distribution.
func (g Gaussian) Sample(params anyvec.Vector, batchSize int) anyvec.Vector {
	c := params.Creator()

	transParams := c.MakeVector(params.Len())
	anyvec.Transpose(params, transParams, params.Len()/2)

	mean := transParams.Slice(0, transParams.Len()/2)
	stddev := transParams.Slice(transParams.Len()/2, transParams.Len()).Copy()

	// sqrt(e^x) = e^(0.5*x)
	stddev.Scale(c.MakeNumeric(0.5))
	anyvec.Exp(stddev)

	noise := c.MakeVector(stddev.Len())
	anyvec.Rand(noise, anyvec.Normal, nil)
	noise.Mul(stddev)
	noise.Add(mean)

	return noise
}

// LogProb computes the output log densities.
func (g Gaussian) LogProb(params anydiff.Res, output anyvec.Vector,
	batchSize int) anydiff.Res {
	c := output.Creator()
	return anydiff.Pool(params, func(params anydiff.Res) anydiff.Res {
		mean, logVariance := g.splitParams(params)
		diffs := anydiff.Square(anydiff.Sub(mean, anydiff.NewConst(output)))
		normalizedDiffs := anydiff.Scale(
			anydiff.Div(diffs, anydiff.Exp(logVariance)),
			c.MakeNumeric(-0.5),
		)
		// ln(1/sqrt(2*pi*s^2)) = -0.5*(ln(2*pi) + ln(s^2))
		log2Pi := c.MakeNumeric(math.Log(2 * math.Pi))
		logNorm := anydiff.Scale(
			anydiff.AddScalar(logVariance, log2Pi),
			c.MakeNumeric(-0.5),
		)
		return anydiff.SumCols(&anydiff.Matrix{
			Data: anydiff.Add(logNorm, normalizedDiffs),
			Rows: batchSize,
			Cols: mean.Output().Len() / batchSize,
		})
	})
}

// KL computes the KL divergences between two batches of
// distributions.
func (g Gaussian) KL(params1, params2 anydiff.Res, batchSize int) anydiff.Res {
	c := params1.Output().Creator()
	return anydiff.Pool(params1, func(params1 anydiff.Res) anydiff.Res {
		return anydiff.Pool(params2, func(params2 anydiff.Res) anydiff.Res {
			mean1, logVar1 := g.splitParams(params1)
			mean2, logVar2 := g.splitParams(params2)
			var1 := anydiff.Exp(logVar1)
			var2 := anydiff.Exp(logVar2)
			return anydiff.SumCols(&anydiff.Matrix{
				// ((u1 - u2)^2 + s1^2 - s2^2) / (2*s2^2) + ln(s2/s1)
				Data: anydiff.Add(
					anydiff.Div(
						anydiff.Add(
							anydiff.Pow(
								anydiff.Sub(mean1, mean2),
								c.MakeNumeric(2),
							),
							anydiff.Sub(var1, var2),
						),
						anydiff.Scale(
							var2,
							c.MakeNumeric(2),
						),
					),
					anydiff.Scale(
						anydiff.Sub(logVar2, logVar1),
						c.MakeNumeric(0.5),
					),
				),
				Rows: batchSize,
				Cols: mean1.Output().Len() / batchSize,
			})
		})
	})
}

func (g Gaussian) splitParams(params anydiff.Res) (mean, logVariance anydiff.Res) {
	halfLen := params.Output().Len() / 2
	mat := &anydiff.Matrix{Data: params, Rows: halfLen, Cols: 2}
	tr := anydiff.Transpose(mat)
	mean = anydiff.Slice(tr.Data, 0, halfLen)
	logVariance = anydiff.Slice(tr.Data, halfLen, halfLen*2)
	return
}

// Tuple is a tuple of action spaces which itself serves
// as an action space.
//
// Each child action space has fixed-length parameter
// vectors, the sizes of which are stored in ParamSizes.
// Also, samples from each child action space must be of
// a fixed size, which is stored in SampleSizes.
//
// Parameter vectors for a tuple are of the form
// <a1, ..., am, b1, ..., bn, ...>, where
// <a1, ..., am> is the parameter vector for the first
// subspace, <b1, ..., bn> for the second, etc.
//
// A Tuple must contain at least one space.
// Empty tuples are not supported.
type Tuple struct {
	Spaces      []interface{}
	ParamSizes  []int
	SampleSizes []int
}

// Sample samples from the tuple elements and returns a
// packed tuple of results.
//
// This panics if a sub-space is not a Sampler.
func (t *Tuple) Sample(params anyvec.Vector, batch int) anyvec.Vector {
	unpacked := unpackTuples(anydiff.NewConst(params), t.ParamSizes, batch)
	var sampled []anyvec.Vector
	for i, subParams := range unpacked {
		sampler := t.Spaces[i].(Sampler)
		sampled = append(sampled, sampler.Sample(subParams.Output(), batch))
	}
	return packTuples(sampled, batch)
}

// LogProb computes the joint probability of the sampled
// output.
//
// This panics if a sub-space is not a LogProber.
func (t *Tuple) LogProb(params anydiff.Res, output anyvec.Vector,
	batch int) anydiff.Res {
	return anydiff.Pool(params, func(params anydiff.Res) anydiff.Res {
		unpackedParams := unpackTuples(params, t.ParamSizes, batch)
		unpackedSamples := unpackTuples(anydiff.NewConst(output), t.SampleSizes, batch)
		var totalProbs anydiff.Res
		for i, samples := range unpackedSamples {
			params := unpackedParams[i]
			logProber := t.Spaces[i].(LogProber)
			logProb := logProber.LogProb(params, samples.Output(), batch)
			if totalProbs == nil {
				totalProbs = logProb
			} else {
				totalProbs = anydiff.Add(totalProbs, logProb)
			}
		}
		return totalProbs
	})
}

// KL computes the KL divergences between two batches of
// distributions.
//
// This panics if a sub-space is not a KLer.
func (t *Tuple) KL(params1, params2 anydiff.Res, batch int) anydiff.Res {
	return anydiff.Pool(params1, func(params1 anydiff.Res) anydiff.Res {
		return anydiff.Pool(params2, func(r anydiff.Res) anydiff.Res {
			unpacked1 := unpackTuples(params1, t.ParamSizes, batch)
			unpacked2 := unpackTuples(params2, t.ParamSizes, batch)
			var totalKL anydiff.Res
			for i, p1 := range unpacked1 {
				p2 := unpacked2[i]
				kler := t.Spaces[i].(KLer)
				kl := kler.KL(p1, p2, batch)
				if totalKL == nil {
					totalKL = kl
				} else {
					totalKL = anydiff.Add(totalKL, kl)
				}
			}
			return totalKL
		})
	})
}

// Entropy computes the entropy for each parameter tuple
// in the batch.
//
// This panics if a sub-space is not an Entropyer.
func (t *Tuple) Entropy(params anydiff.Res, batch int) anydiff.Res {
	return anydiff.Pool(params, func(params anydiff.Res) anydiff.Res {
		unpacked := unpackTuples(params, t.ParamSizes, batch)
		var totalEntropy anydiff.Res
		for i, subParams := range unpacked {
			entropyer := t.Spaces[i].(Entropyer)
			ent := entropyer.Entropy(subParams, batch)
			if totalEntropy == nil {
				totalEntropy = ent
			} else {
				totalEntropy = anydiff.Add(totalEntropy, ent)
			}
		}
		return totalEntropy
	})
}

func batchedDot(vecs1, vecs2 anydiff.Res, batchSize int) anydiff.Res {
	products := anydiff.Mul(vecs1, vecs2)
	return anydiff.SumCols(&anydiff.Matrix{
		Data: products,
		Rows: batchSize,
		Cols: vecs1.Output().Len() / batchSize,
	})
}

// sampleProbabilities samples a one-hot vector from a
// list of index probabilities.
func sampleProbabilities(p anyvec.Vector) anyvec.Vector {
	randNum := rand.Float64()
	idx := p.Len() - 1
	switch data := p.Data().(type) {
	case []float32:
		for i, x := range data {
			randNum -= float64(x)
			if randNum < 0 {
				idx = i
				break
			}
		}
	case []float64:
		for i, x := range data {
			randNum -= x
			if randNum < 0 {
				idx = i
				break
			}
		}
	default:
		panic(fmt.Sprintf("cannot sample from %T", data))
	}

	oneHot := make([]float64, p.Len())
	oneHot[idx] = 1
	return p.Creator().MakeVectorData(p.Creator().MakeNumericList(oneHot))
}

// pairWithComplement turns a vector of the form
//
//     <a, b, c, ...>
//
// into a vector of the form
//
//     <1-a, a, 1-b, b, 1-c, c, ...>
func pairWithComplement(in anydiff.Res) anydiff.Res {
	return anydiff.Pool(in, func(in anydiff.Res) anydiff.Res {
		return sideBySide(anydiff.Complement(in), in)
	})
}

// sideBySide turns two vectors
//
//     <a1, b1, c1, ...>
//     <a2, b2, c2, ...>
//
// into a vector of the form
//
//     <a1, a2, b1, b2, c1, c2, ...>
func sideBySide(v1, v2 anydiff.Res) anydiff.Res {
	return anydiff.Transpose(&anydiff.Matrix{
		Data: anydiff.Concat(v1, v2),
		Rows: 2,
		Cols: v1.Output().Len(),
	}).Data
}

// unpackTuple sseparates vectors in a batch of packed
// vector tuples.
//
// You should pool params before calling this.
func unpackTuples(params anydiff.Res, vecSizes []int, batch int) []anydiff.Res {
	totalSize := 0
	for _, s := range vecSizes {
		totalSize += s
	}

	if params.Output().Len() != totalSize*batch {
		panic(fmt.Sprintf("param size should be %d but got %d", totalSize*batch,
			params.Output().Len()))
	}

	unjoined := make([][]anydiff.Res, len(vecSizes))
	var offset int
	for i := 0; i < batch; i++ {
		for spaceIdx, size := range vecSizes {
			subVec := anydiff.Slice(params, offset, offset+size)
			offset += size
			unjoined[spaceIdx] = append(unjoined[spaceIdx], subVec)
		}
	}

	joined := make([]anydiff.Res, len(vecSizes))
	for i, list := range unjoined {
		if len(list) == 0 {
			joined[i] = anydiff.NewConst(params.Output().Creator().MakeVector(0))
		} else {
			joined[i] = anydiff.Concat(list...)
		}
	}

	return joined
}

// packTuples does the inverse of unpackTuples.
func packTuples(eachPacked []anyvec.Vector, batch int) anyvec.Vector {
	if batch == 0 {
		return eachPacked[0].Creator().MakeVector(0)
	}
	var eachSplit [][]anyvec.Vector
	for _, packed := range eachPacked {
		chunkSize := packed.Len() / batch
		var split []anyvec.Vector
		for i := 0; i < batch; i++ {
			s := packed.Slice(i*chunkSize, (i+1)*chunkSize)
			split = append(split, s)
		}
		eachSplit = append(eachSplit, split)
	}
	var unjoined []anyvec.Vector
	for i := 0; i < batch; i++ {
		for _, split := range eachSplit {
			unjoined = append(unjoined, split[i])
		}
	}
	return eachPacked[0].Creator().Concat(unjoined...)
}
