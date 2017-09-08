package anyrl

import (
	"sync"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/lazyseq"
)

// TapeMaker is a function which generates a tape and a
// channel for writing to that tape.
//
// See lazyseq.ReferenceTape for an example.
type TapeMaker func(c anyvec.Creator) (tape lazyseq.Tape,
	writer chan<- *anyseq.Batch)

// RNNRoller runs RNN agents through environments and
// saves the results to RolloutSets.
type RNNRoller struct {
	Block       anyrnn.Block
	ActionSpace Sampler

	// Creator is used to convert observations to and
	// from the block.
	// If nil, the creator of the block's first parameter
	// is used.
	Creator anyvec.Creator

	// These functions are called to produce tapes when
	// building a RolloutSet.
	//
	// You can set these in order to use special storage
	// techniques (e.g. compression or on-disk storage).
	//
	// For nil fields, lazyseq.ReferenceTape is used.
	MakeInputTape    TapeMaker
	MakeActionTape   TapeMaker
	MakeAgentOutTape TapeMaker
}

// Rollout produces one rollout per environment.
func (r *RNNRoller) Rollout(envs ...Env) (rollouts *RolloutSet, err error) {
	defer essentials.AddCtxTo("rollout RNN", &err)

	c := r.creator()
	inputs, inputCh := makeTape(c, r.MakeInputTape)
	actions, actionCh := makeTape(c, r.MakeActionTape)
	agentOuts, agentOutCh := makeTape(c, r.MakeAgentOutTape)

	defer func() {
		close(inputCh)
		close(actionCh)
		close(agentOutCh)
	}()

	rewards, err := r.rolloutChans(inputCh, actionCh, agentOutCh, envs)
	if err != nil {
		return nil, err
	}

	return &RolloutSet{
		Inputs:    inputs,
		Actions:   actions,
		AgentOuts: agentOuts,
		Rewards:   rewards,
	}, nil
}

func (r *RNNRoller) rolloutChans(inputCh, actionCh, agentOutCh chan<- *anyseq.Batch,
	envs []Env) (Rewards, error) {
	if len(envs) == 0 {
		return nil, nil
	}

	initBatch, err := rolloutReset(r.creator(), envs)
	if err != nil {
		return nil, err
	}
	rewards := make(Rewards, len(initBatch.Present))

	inBatch := initBatch
	state := r.Block.Start(len(initBatch.Present))
	for inBatch.NumPresent() > 0 {
		inputCh <- inBatch

		if inBatch.NumPresent() < state.Present().NumPresent() {
			state = state.Reduce(inBatch.Present)
		}
		blockRes := r.Block.Step(state, inBatch.Packed)
		state = blockRes.State()

		out := r.ActionSpace.Sample(blockRes.Output(), inBatch.NumPresent())
		actionBatch := &anyseq.Batch{Packed: out, Present: inBatch.Present}

		actionCh <- actionBatch
		agentOutCh <- &anyseq.Batch{Packed: blockRes.Output(), Present: inBatch.Present}

		var rewardBatch []float64
		inBatch, rewardBatch, err = rolloutStep(actionBatch, envs)
		if err != nil {
			return nil, err
		}

		for i, pres := range actionBatch.Present {
			if pres {
				rewards[i] = append(rewards[i], rewardBatch[0])
				rewardBatch = rewardBatch[1:]
			}
		}
	}

	return rewards, nil
}

func (r *RNNRoller) creator() anyvec.Creator {
	if r.Creator != nil {
		return r.Creator
	} else {
		return anynet.AllParameters(r.Block)[0].Output().Creator()
	}
}

func rolloutReset(c anyvec.Creator, envs []Env) (*anyseq.Batch, error) {
	initBatch := &anyseq.Batch{
		Present: make([]bool, len(envs)),
	}

	var allObs []float64
	for i, e := range envs {
		obs, err := e.Reset()
		if err != nil {
			return nil, err
		}
		initBatch.Present[i] = true
		allObs = append(allObs, obs...)
	}

	initBatch.Packed = anyvec.Make(c, allObs)

	return initBatch, nil
}

func rolloutStep(actions *anyseq.Batch, envs []Env) (obs *anyseq.Batch,
	rewards []float64, err error) {
	c := actions.Packed.Creator()
	obs = &anyseq.Batch{
		Present: make([]bool, len(actions.Present)),
	}
	var splitActions [][]float64
	var presentEnvs []Env

	actionChunkSize := actions.Packed.Len() / actions.NumPresent()
	var actionOffset int
	actionSlice := c.Float64Slice(actions.Packed.Data())
	for i, pres := range actions.Present {
		if pres {
			action := actionSlice[actionOffset : actionOffset+actionChunkSize]
			actionOffset += actionChunkSize
			splitActions = append(splitActions, action)
			presentEnvs = append(presentEnvs, envs[i])
		}
	}

	obsVecs, rewards, dones, errs := batchStep(presentEnvs, splitActions)

	var presentIdx int
	var joinObs []float64
	for i, pres := range actions.Present {
		if !pres {
			continue
		}
		obsVec, done, err := obsVecs[presentIdx], dones[presentIdx], errs[presentIdx]
		presentIdx++
		if err != nil {
			return nil, nil, err
		}
		if !done {
			obs.Present[i] = true
			joinObs = append(joinObs, obsVec...)
		}
	}

	obs.Packed = anyvec.Make(c, joinObs)

	return
}

func batchStep(envs []Env, actions [][]float64) (obs [][]float64,
	rewards []float64, done []bool, err []error) {
	obs = make([][]float64, len(envs))
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

func makeTape(c anyvec.Creator, maker TapeMaker) (lazyseq.Tape, chan<- *anyseq.Batch) {
	if maker != nil {
		return maker(c)
	} else {
		return lazyseq.ReferenceTape(c)
	}
}
