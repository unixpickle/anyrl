package anypg

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyrl"
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
