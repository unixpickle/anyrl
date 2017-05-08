package anypg

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
)

// A Regularizer regularizes the actions taken by a policy
// by encouraging exploration.
type Regularizer interface {
	// Regularize produces a regularization term for the
	// policy gradient objective.
	// It takes a batch of action space parameters and
	// produces, for each batch element, a regularization
	// term to be maximized.
	Regularize(actionParams anydiff.Res, batchSize int) anydiff.Res
}

// EntropyReg implements entropy regularization by
// encouraging action distributions with high entropy.
type EntropyReg struct {
	Entropyer anyrl.Entropyer

	// Coeff controls the strength of the regularizer.
	// A value of 0.01 is a good starting point.
	Coeff float64
}

// Regularize produces a scaled entropy term.
func (e *EntropyReg) Regularize(params anydiff.Res, batchSize int) anydiff.Res {
	c := params.Output().Creator()
	return anydiff.Scale(
		e.Entropyer.Entropy(params, batchSize),
		c.MakeNumeric(e.Coeff),
	)
}

// InvEntropyReg uses the negative reciprocal of a
// distribution's entropy as a regularization term.
// This way, as the entropy approaches zero, the term
// approaches -infinity.
type InvEntropyReg struct {
	Entropyer anyrl.Entropyer

	// Coeff controls the strength of the regularizer.
	Coeff float64
}

// Regularize produces a scaled inverse entropy term.
func (i *InvEntropyReg) Regularize(params anydiff.Res, batchSize int) anydiff.Res {
	c := params.Output().Creator()
	entropy := i.Entropyer.Entropy(params, batchSize)
	recip := anydiff.Pow(entropy, c.MakeNumeric(-1))
	return anydiff.Scale(recip, c.MakeNumeric(-i.Coeff))
}

// AverageReg computes the average regularization term
// across all rollouts.
func AverageReg(c anyvec.Creator, agentOuts lazyseq.Tape,
	reg Regularizer) anyvec.Numeric {
	inSeq := lazyseq.TapeRereader(c, agentOuts)
	regSeq := lazyseq.Map(inSeq, reg.Regularize)
	return anyvec.Sum(lazyseq.Mean(regSeq).Output())
}
