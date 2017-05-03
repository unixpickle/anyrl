package anya3c

import (
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
)

// A worker manages the current state of a worker.
type worker struct {
	Agent *LocalAgent
	Env   anyrl.Env

	// Results from latest StepEnv or Reset.
	EnvObs  anyvec.Vector
	EnvDone bool

	// AgentRes is the result of applying the RNN to the
	// current EnvObs.
	//
	// It may be nil.
	//
	// Order: base, actor, critic.
	AgentRes []anyrnn.Res

	// AgentState is the state from the most recent RNN
	// evaluation (or reset).
	//
	// Order: base, actor, critic.
	AgentState []anyrnn.State

	// Tracks the total undiscounted reward for the
	// current episode.
	RewardSum float64
}

// Reset resets the environment and the RNN state.
func (w *worker) Reset() error {
	var err error
	w.EnvObs, err = w.Env.Reset()
	if err != nil {
		return err
	}
	w.EnvDone = false
	for i, block := range w.blocks() {
		w.AgentState[i] = block.Start(1)
	}
	w.AgentRes = nil
	w.RewardSum = 0
	return nil
}

// StepAgent applies the agent to the latest observation
// if it has not already been applied.
func (w *worker) StepAgent() {
	if w.AgentRes != nil {
		return
	}
	blocks := w.blocks()
	baseOut := blocks[0].Step(w.AgentState[0], w.EnvObs)
	actorOut := blocks[1].Step(w.AgentState[1], baseOut.Output())
	criticOut := blocks[2].Step(w.AgentState[2], baseOut.Output())
	w.AgentRes = []anyrnn.Res{baseOut, actorOut, criticOut}
	for i, res := range w.AgentRes {
		w.AgentState[i] = res.State()
	}
}

// StepEnv uses the latest agent result to sample an
// action and take a step in the environment.
//
// You should call StepAgent before calling StepEnv.
func (w *worker) StepEnv() (reward float64, action anyvec.Vector, err error) {
	actorOut := w.AgentRes[1].Output()
	action = w.Agent.ActionSpace.Sample(actorOut, 1)
	w.EnvObs, reward, w.EnvDone, err = w.Env.Step(action)
	if err != nil {
		return
	}
	w.RewardSum += reward
	w.AgentRes = nil
	return
}

// blocks returns the policy blocks in the order of
// w.AgentRes and w.AgentState.
func (w *worker) blocks() []anyrnn.Block {
	return []anyrnn.Block{w.Agent.Base, w.Agent.Actor, w.Agent.Critic}
}
