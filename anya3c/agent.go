package anya3c

import (
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
