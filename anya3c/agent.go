package anya3c

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

// An ActionSpace is used to sample and propagate through
// action parameters produced by an actor.
type ActionSpace interface {
	anyrl.Sampler
	anyrl.LogProber
}

// Agent is a set of RNNs used to implement the actor and
// the critic.
//
// Inputs are all fed into Base.
// The output of Base is fed into Actor and Critic, which
// implement the actor and the critic respectively.
// The output of the Actor is used via ActionSpace.
//
// The RNN blocks must work with serializer.Copy, since
// local copies of an agent are made for each worker.
type Agent struct {
	Base, Actor, Critic anyrnn.Block
	ActionSpace         ActionSpace
}

// AllParameters finds all of the agent's parameters via
// anynet.AllParameters.
func (a *Agent) AllParameters() []*anydiff.Var {
	return anynet.AllParameters(a.Base, a.Actor, a.Critic)
}

// Copy produces a copy of the agent.
func (a *Agent) Copy() (*Agent, error) {
	res := &Agent{ActionSpace: a.ActionSpace}
	srcBlocks := []anyrnn.Block{a.Base, a.Actor, a.Critic}
	dstBlocks := []*anyrnn.Block{&res.Base, &res.Actor, &res.Critic}
	for i, src := range srcBlocks {
		copied, err := serializer.Copy(src)
		if err != nil {
			name := []string{"base", "actor", "critic"}
			return nil, essentials.AddCtx("copy agent "+name[i], err)
		}
		*dstBlocks[i] = copied.(anyrnn.Block)
	}
	return res, nil
}

// A LocalAgent is a local copy of a global agent.
type LocalAgent struct {
	*Agent

	// Params indicates which parameters in the RNNs to
	// optimize.
	//
	// The order here matters, as it makes it possible
	// to map between global and local parameters.
	Params []*anydiff.Var
}
