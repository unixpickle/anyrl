package trpo

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyrnn"
)

// A RolloutSet is a batch of rollouts.
//
// An instance of RolloutSet contains three different
// sequence batches, each describing a different aspect
// of the episodes.
//
// All of the sequence batches are constant, i.e. nothing
// will happen if they are back-propagated through.
type RolloutSet struct {
	// Inputs contains the inputs to the agent at
	// each timestep.
	Inputs lazyrnn.Rereader

	// Rewards contains the immediate reward at each
	// timestep.
	Rewards lazyrnn.Rereader

	// SampledOuts contains the sampled agent action
	// at each timestep.
	SampledOuts lazyrnn.Rereader
}

// PackRolloutSets joins multiple RolloutSets into one
// larger set.
func PackRolloutSets(c anyvec.Creator, rs []*RolloutSet) *RolloutSet {
	res := &RolloutSet{}
	fieldGetters := []func(r *RolloutSet) *lazyrnn.Rereader{
		func(r *RolloutSet) *lazyrnn.Rereader {
			return &r.Inputs
		},
		func(r *RolloutSet) *lazyrnn.Rereader {
			return &r.Rewards
		},
		func(r *RolloutSet) *lazyrnn.Rereader {
			return &r.SampledOuts
		},
	}
	for _, getter := range fieldGetters {
		var seqs []lazyrnn.Rereader
		for _, r := range rs {
			seqs = append(seqs, *getter(r))
		}
		*getter(res) = lazyrnn.PackRereader(c, seqs)
	}
	return res
}
