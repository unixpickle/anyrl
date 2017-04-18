package anyrl

import (
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
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

// RolloutRNN performs rollouts using an RNN.
//
// If feedback is true, the reward from the previous
// time-step is appended to the end of each observation.
func RolloutRNN(c anyvec.Creator, agent anyrnn.Block, actionSampler Sampler,
	envs ...Env) (*RolloutSet, error) {
	// TODO: support other Tape types.
	inputs, inputCh := lazyrnn.ReferenceTape()
	rewards, rewardsCh := lazyrnn.ReferenceTape()
	sampled, sampledCh := lazyrnn.ReferenceTape()

	defer func() {
		close(inputCh)
		close(rewardsCh)
		close(sampledCh)
	}()

	res := &RolloutSet{Inputs: inputs, Rewards: rewards, SampledOuts: sampled}

	if len(envs) == 0 {
		return res, nil
	}

	initBatch, err := rolloutReset(c, envs)
	if err != nil {
		return nil, err
	}

	inBatch := initBatch
	state := agent.Start(len(initBatch.Present))
	for inBatch.NumPresent() > 0 {
		inputCh <- inBatch

		if inBatch.NumPresent() < state.Present().NumPresent() {
			state = state.Reduce(inBatch.Present)
		}
		blockRes := agent.Step(state, inBatch.Packed)
		state = blockRes.State()

		out := actionSampler.Sample(blockRes.Output(), inBatch.NumPresent())
		actionBatch := &anyseq.Batch{Packed: out, Present: inBatch.Present}

		sampledCh <- actionBatch

		var rewardBatch *anyseq.Batch
		inBatch, rewardBatch, err = rolloutStep(actionBatch, envs)
		if err != nil {
			return nil, err
		}

		rewardsCh <- rewardBatch
	}

	return res, nil
}

func rolloutReset(c anyvec.Creator, envs []Env) (*anyseq.Batch, error) {
	initBatch := &anyseq.Batch{
		Present: make([]bool, len(envs)),
		Packed:  c.MakeVector(0),
	}

	for i, e := range envs {
		obs, err := e.Reset()
		if err != nil {
			return nil, err
		}
		initBatch.Present[i] = true
		initBatch.Packed = c.Concat(initBatch.Packed, obs)
	}

	return initBatch, nil
}

func rolloutStep(actions *anyseq.Batch, envs []Env) (obs, rewards *anyseq.Batch,
	err error) {
	c := actions.Packed.Creator()
	obs = &anyseq.Batch{
		Present: make([]bool, len(actions.Present)),
		Packed:  c.MakeVector(0),
	}
	rewards = &anyseq.Batch{
		Present: actions.Present,
		Packed:  c.MakeVector(0),
	}
	actionChunkSize := actions.Packed.Len() / actions.NumPresent()
	var actionOffset int
	for i, e := range envs {
		if !actions.Present[i] {
			continue
		}
		action := actions.Packed.Slice(actionOffset, actionOffset+actionChunkSize)
		actionOffset += actionChunkSize
		obsVec, rew, done, err := e.Step(action)
		if err != nil {
			return nil, nil, err
		}
		if !done {
			obs.Present[i] = true
			obs.Packed = c.Concat(obs.Packed, obsVec)
		}
		rewVec := c.MakeVector(1)
		rewVec.AddScalar(c.MakeNumeric(rew))
		rewards.Packed = c.Concat(rewards.Packed, rewVec)
	}
	return
}
