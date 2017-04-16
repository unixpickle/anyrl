package trpo

import "github.com/unixpickle/lazyrnn"

// A RolloutBatch is a batch of rollouts.
//
// A RolloutBatch contains three different sequence
// batches, each describing a different aspect of each
// timestep of each episode.
//
// All of the sequence batches are constant, i.e. nothing
// will happen if they are back-propagated through.
type RolloutBatch struct {
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

// JoinRolloutBatches joins multiple RolloutBatches into
// a single, larger batch.
func JoinRolloutBatches(batches ...*RolloutBatch) *RolloutBatch {
	// TODO: this.
	panic("not yet implemented")
}
