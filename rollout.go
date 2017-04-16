package trpo

import "github.com/unixpickle/lazyrnn"

// A RolloutSet is a batch of recorded episodes.
//
// An instance of RolloutSet contains three different
// tapes, each describing a different aspect of the
// episode.
type RolloutSet struct {
	// Inputs contains the inputs to the agent at
	// each timestep.
	Inputs lazyrnn.Tape

	// Rewards contains the immediate reward at each
	// timestep.
	Rewards lazyrnn.Tape

	// SampledOuts contains the sampled agent action
	// at each timestep.
	SampledOuts lazyrnn.Tape
}

// PackRolloutSets joins multiple RolloutSets into one
// larger set.
func PackRolloutSets(rs []*RolloutSet) *RolloutSet {
	res := &RolloutSet{}
	fieldGetters := []func(r *RolloutSet) *lazyrnn.Tape{
		func(r *RolloutSet) *lazyrnn.Tape {
			return &r.Inputs
		},
		func(r *RolloutSet) *lazyrnn.Tape {
			return &r.Rewards
		},
		func(r *RolloutSet) *lazyrnn.Tape {
			return &r.SampledOuts
		},
	}
	for _, getter := range fieldGetters {
		var tapes []lazyrnn.Tape
		for _, r := range rs {
			tapes = append(tapes, *getter(r))
		}
		*getter(res) = lazyrnn.PackTape(tapes)
	}
	return res
}
