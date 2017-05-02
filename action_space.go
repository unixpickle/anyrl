package anyrl

import (
	"fmt"
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
	diff := anydiff.Sub(offOn1, offOn2)
	probs1 := anydiff.Exp(offOn1)
	return batchedDot(probs1, diff, batchSize)
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

func batchedDot(vecs1, vecs2 anydiff.Res, batchSize int) anydiff.Res {
	products := anydiff.Mul(vecs1, vecs2)
	return anydiff.SumCols(&anydiff.Matrix{
		Data: products,
		Rows: batchSize,
		Cols: vecs1.Output().Len() / batchSize,
	})
}

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
