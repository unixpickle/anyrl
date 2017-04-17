package anyrl

import (
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/lazyrnn"
)

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

// RemainingRewards derives a tape from r.Rewards which,
// at each time-step, has the total reward from that
// time-step to the end of the episode.
func (r *RolloutSet) RemainingRewards() lazyrnn.Tape {
	var sum *anyseq.Batch
	for batch := range r.Rewards.ReadTape(0, -1) {
		if sum == nil {
			sum = &anyseq.Batch{
				Present: batch.Present,
				Packed:  batch.Packed.Copy(),
			}
		} else {
			sum.Packed.Add(batch.Expand(sum.Present).Packed)
		}
	}

	resTape, writer := lazyrnn.ReferenceTape()

	for batch := range r.Rewards.ReadTape(0, -1) {
		// Create two separate copies of the sum to
		// avoid modifying the batch we send.
		writer <- sum.Reduce(batch.Present)
		sum = sum.Reduce(batch.Present)
		sum.Packed.Sub(batch.Packed)
	}

	close(writer)
	return resTape
}
