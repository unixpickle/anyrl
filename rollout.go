package anyrl

import (
	"sync"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
)

// A RolloutSet is a batch of recorded episodes.
//
// An instance of RolloutSet contains three different
// tapes, each describing a different aspect of the
// episode.
type RolloutSet struct {
	// Inputs contains the inputs to the agent at
	// each timestep.
	Inputs lazyseq.Tape

	// Rewards contains the immediate reward at each
	// timestep.
	Rewards lazyseq.Tape

	// SampledOuts contains the sampled agent action
	// at each timestep.
	SampledOuts lazyseq.Tape
}

// PackRolloutSets joins multiple RolloutSets into one
// larger set.
func PackRolloutSets(rs []*RolloutSet) *RolloutSet {
	res := &RolloutSet{}
	fieldGetters := []func(r *RolloutSet) *lazyseq.Tape{
		func(r *RolloutSet) *lazyseq.Tape {
			return &r.Inputs
		},
		func(r *RolloutSet) *lazyseq.Tape {
			return &r.Rewards
		},
		func(r *RolloutSet) *lazyseq.Tape {
			return &r.SampledOuts
		},
	}
	for _, getter := range fieldGetters {
		var tapes []lazyseq.Tape
		for _, r := range rs {
			tapes = append(tapes, *getter(r))
		}
		*getter(res) = lazyseq.PackTape(tapes)
	}
	return res
}

// TotalRewards sums the rewards for each rollout.
func (r *RolloutSet) TotalRewards(c anyvec.Creator) anyvec.Vector {
	return TotalRewards(c, r.Rewards)
}

// MeanReward sums the rewards for each rollout, then
// computes the mean of the sums.
func (r *RolloutSet) MeanReward(c anyvec.Creator) anyvec.Numeric {
	return MeanReward(c, r.Rewards)
}

// RolloutRNN performs rollouts using an RNN.
//
// One rollout is performed per environment.
func RolloutRNN(c anyvec.Creator, agent anyrnn.Block, actionSampler Sampler,
	envs ...Env) (*RolloutSet, error) {
	inputs, inputCh := lazyseq.ReferenceTape()
	rewards, rewardsCh := lazyseq.ReferenceTape()
	sampled, sampledCh := lazyseq.ReferenceTape()

	defer func() {
		close(inputCh)
		close(rewardsCh)
		close(sampledCh)
	}()

	err := RolloutRNNChans(c, agent, actionSampler, inputCh, rewardsCh, sampledCh, envs...)
	if err != nil {
		return nil, err
	}
	return &RolloutSet{Inputs: inputs, Rewards: rewards, SampledOuts: sampled}, nil
}

// RolloutRNNChans is like RolloutRNN, but the batches of
// sampled data are sent directly to the given channels.
// This makes it possible to use custom tape types.
//
// RolloutRNNChans will not close the channels after it is
// done with them.
func RolloutRNNChans(c anyvec.Creator, agent anyrnn.Block, actionSampler Sampler,
	inputCh, rewardsCh, sampledCh chan<- *anyseq.Batch, envs ...Env) error {
	if len(envs) == 0 {
		return nil
	}

	initBatch, err := rolloutReset(c, envs)
	if err != nil {
		return err
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
			return err
		}

		rewardsCh <- rewardBatch
	}

	return nil
}

func rolloutReset(c anyvec.Creator, envs []Env) (*anyseq.Batch, error) {
	initBatch := &anyseq.Batch{
		Present: make([]bool, len(envs)),
	}

	var allObs []anyvec.Vector
	for i, e := range envs {
		obs, err := e.Reset()
		if err != nil {
			return nil, err
		}
		initBatch.Present[i] = true
		allObs = append(allObs, obs)
	}

	initBatch.Packed = c.Concat(allObs...)

	return initBatch, nil
}

func rolloutStep(actions *anyseq.Batch, envs []Env) (obs, rewards *anyseq.Batch,
	err error) {
	c := actions.Packed.Creator()
	obs = &anyseq.Batch{
		Present: make([]bool, len(actions.Present)),
	}
	rewards = &anyseq.Batch{
		Present: actions.Present,
	}
	actionChunkSize := actions.Packed.Len() / actions.NumPresent()
	var actionOffset int
	var splitActions []anyvec.Vector
	var presentEnvs []Env
	for i, pres := range actions.Present {
		if pres {
			action := actions.Packed.Slice(actionOffset, actionOffset+actionChunkSize)
			actionOffset += actionChunkSize
			splitActions = append(splitActions, action)
			presentEnvs = append(presentEnvs, envs[i])
		}
	}

	obsVecs, rews, dones, errs := batchStep(presentEnvs, splitActions)

	var presentIdx int
	var joinObs []anyvec.Vector
	var rewData []float64
	for i, pres := range actions.Present {
		if !pres {
			continue
		}
		obsVec, rew, done, err := obsVecs[presentIdx], rews[presentIdx],
			dones[presentIdx], errs[presentIdx]
		presentIdx++
		if err != nil {
			return nil, nil, err
		}
		if !done {
			obs.Present[i] = true
			joinObs = append(joinObs, obsVec)
		}
		rewData = append(rewData, rew)
	}

	obs.Packed = c.Concat(joinObs...)
	rewards.Packed = c.MakeVectorData(c.MakeNumericList(rewData))

	return
}

type envStepRes struct {
	observation anyvec.Vector
	reward      float64
	done        bool
	err         error
}

func batchStep(envs []Env, actions []anyvec.Vector) (obs []anyvec.Vector,
	rewards []float64, done []bool, err []error) {
	obs = make([]anyvec.Vector, len(envs))
	rewards = make([]float64, len(envs))
	done = make([]bool, len(envs))
	err = make([]error, len(envs))
	var wg sync.WaitGroup
	for i, e := range envs {
		wg.Add(1)
		go func(i int, e Env) {
			defer wg.Done()
			obs[i], rewards[i], done[i], err[i] = e.Step(actions[i])
		}(i, e)
	}
	wg.Wait()
	return
}
