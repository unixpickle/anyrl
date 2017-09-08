package anyrl

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
)

// A RolloutSet is a batch of recorded episodes.
type RolloutSet struct {
	// Inputs contains the inputs to the agent at
	// each timestep.
	Inputs lazyseq.Tape

	// Actions contains the actions taken by the agent
	// at each timestep.
	Actions lazyseq.Tape

	// Rewards contains the rewards given to the agent
	// at each timestep.
	Rewards Rewards

	// AgentOuts contains the raw outputs from the
	// agent at each timestep.
	//
	// This can be nil if the agent does not produce
	// vector values at each timestep.
	// This field is mostly meant for agents which are
	// based on function approximators.
	AgentOuts lazyseq.Tape
}

// PackRolloutSets joins multiple RolloutSets into one
// larger set.
func PackRolloutSets(c anyvec.Creator, rs []*RolloutSet) *RolloutSet {
	res := &RolloutSet{}

	fieldGetters := []func(r *RolloutSet) *lazyseq.Tape{
		func(r *RolloutSet) *lazyseq.Tape {
			return &r.Inputs
		},
		func(r *RolloutSet) *lazyseq.Tape {
			return &r.Actions
		},
		func(r *RolloutSet) *lazyseq.Tape {
			return &r.AgentOuts
		},
	}
	for _, getter := range fieldGetters {
		var tapes []lazyseq.Tape
		for _, r := range rs {
			tapes = append(tapes, *getter(r))
		}

		// Deal with AgentOuts being nil.
		if len(tapes) != 0 && tapes[0] == nil {
			continue
		}

		*getter(res) = lazyseq.PackTape(c, tapes)
	}

	rewards := make([]Rewards, len(rs))
	for i, r := range rs {
		rewards[i] = r.Rewards
	}
	res.Rewards = PackRewards(rewards)

	return res
}

// NumSteps counts the total number of timesteps across
// every episode.
func (r *RolloutSet) NumSteps() int {
	var count int
	for _, seq := range r.Rewards {
		count += len(seq)
	}
	return count
}
