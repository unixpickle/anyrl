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
	numSeqs := len(r.Rewards)
	numSelected := int(math.Ceil(f.Frac * float64(numSeqs)))
	indices := rand.Perm(numSeqs)[:numSelected]
	present := make([]bool, numSeqs)
	for _, j := range indices {
		present[j] = true
	}
	res := &RolloutSet{
		Inputs:  lazyseq.ReduceTape(r.Inputs, present),
		Actions: lazyseq.ReduceTape(r.Actions, present),
		Rewards: r.Rewards.Reduce(present),
	}
	if r.AgentOuts != nil {
		res.AgentOuts = lazyseq.ReduceTape(r.AgentOuts, present)
	}
	return res
}
