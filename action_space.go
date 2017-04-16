package trpo

import (
	"fmt"
	"math/rand"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

// ActionSpace is a parameterized action space probability
// distribution.
//
// For an example, see Softmax.
type ActionSpace interface {
	// Sample samples an action from the action space
	// given the distribution parameters.
	Sample(params anyvec.Vector) anyvec.Vector

	// LogProb produces, for each parameter-output pair
	// in the batch, a log-probability of the parameters
	// producing that output.
	//
	// For continuous distributions, this is the log of
	// the density rather than of the probability.
	LogProb(params anydiff.Res, output anyvec.Vector,
		batchSize int) anydiff.Res

	// KL computes the KL divergence between action space
	// distributions, given the parameters for each.
	//
	// This is batched, just like LogProb.
	// It produces one value per entry in the batch.
	KL(params1, params2 anydiff.Res, batchSize int) anydiff.Res
}

// Softmax is an ActionSpace which applies the softmax
// function to obtain a discrete probability distribution.
// It produces one-hot vector samples.
type Softmax struct{}

// Sample samples a one-hot vector from the softmax
// distribution.
func (s Softmax) Sample(params anyvec.Vector) anyvec.Vector {
	p := params.Copy()
	anyvec.LogSoftmax(p, 0)
	anyvec.Exp(p)

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

func batchedDot(vecs1, vecs2 anydiff.Res, batchSize int) anydiff.Res {
	products := anydiff.Mul(vecs1, vecs2)
	return anydiff.SumCols(&anydiff.Matrix{
		Data: products,
		Rows: batchSize,
		Cols: vecs1.Output().Len() / batchSize,
	})
}
