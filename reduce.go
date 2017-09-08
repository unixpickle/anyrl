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

	// These TapeMakers are called to produce caches of
	// the reduced tapes.
	// If a TapeMaker is nil, no cache is used for the
	// corresponding tape.
	//
	// You may want to set these if you plan to use a
	// reduced tape more than once, since it may be more
	// efficient to cache a reduced version of the tape
	// than to keep re-reading the original tape and
	// reducing it on the fly.
	MakeInputTape    TapeMaker
	MakeActionTape   TapeMaker
	MakeAgentOutTape TapeMaker
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
		Inputs:  reduceTape(f.MakeInputTape, r.Inputs, present),
		Actions: reduceTape(f.MakeActionTape, r.Actions, present),
		Rewards: r.Rewards.Reduce(present),
	}
	if r.AgentOuts != nil {
		res.AgentOuts = reduceTape(f.MakeAgentOutTape, r.AgentOuts, present)
	}
	return res
}

func reduceTape(maker TapeMaker, t lazyseq.Tape, present []bool) lazyseq.Tape {
	reduced := lazyseq.ReduceTape(t, present)
	if maker == nil {
		return reduced
	} else {
		copiedTape, writer := maker(reduced.Creator())
		go func() {
			defer close(writer)
			for batch := range lazyseq.ReduceTape(t, present).ReadTape(0, -1) {
				writer <- batch
			}
		}()
		return copiedTape
	}
}
