package anyrl

import (
	"math"
	"math/rand"

	"github.com/unixpickle/lazyseq"
)

// FracReducer reduces RolloutSets by randomly selecting
// a fraction of the rollouts.
type FracReducer struct {
	Frac float64
}

// Reduce reduces the set of rollouts.
//
// The number of rollouts is always rounded up to avoid
// selecting 0 rollouts.
func (f *FracReducer) Reduce(r *RolloutSet) *RolloutSet {
	first, ok := <-r.Rewards.ReadTape(0, 1)
	if !ok {
		return r
	}
	num := int(math.Ceil(f.Frac * float64(len(first.Present))))
	indices := rand.Perm(len(first.Present))[:num]
	present := make([]bool, len(first.Present))
	for _, j := range indices {
		present[j] = true
	}
	return &RolloutSet{
		Inputs:      lazyseq.ReduceTape(r.Inputs, present),
		Rewards:     lazyseq.ReduceTape(r.Rewards, present),
		SampledOuts: lazyseq.ReduceTape(r.SampledOuts, present),
	}
}
