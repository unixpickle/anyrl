package anya3c

import (
	"errors"
	"fmt"
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

// Returned when a parameter server call fails because it
// is already closed.
var errClosed = errors.New("parameter server is closed")

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

// A ParamServer manages a shared set of parameters.
type ParamServer interface {
	// LocalCopy creates a copy of the global agent
	// in a thread-safe manner.
	LocalCopy() (*LocalAgent, error)

	// Sync updates the local parameters to reflect the
	// latest parameters.
	Sync(l *LocalAgent) error

	// Update updates the global parameters based on the
	// gradient from a local agent.
	//
	// Once Update receives a gradient, it owns that
	// gradient forever.
	// It may modify the gradient, reuse the gradient, etc.
	//
	// This may block, but it may also be asynchronous.
	Update(g anydiff.Grad, l *LocalAgent) error

	// Close terminates the server and cleans up any
	// resources associated with it.
	//
	// Close will block until it is no longer updating the
	// global policy on any Goroutine.
	Close() error
}

// paramServer is a ParamServer that stores the global
// parameters as references and uses synchronization
// primitives to control access to them.
type paramServer struct {
	StepSize float64
	Agent    *Agent

	Params     []*anydiff.Var
	Locks      []*sync.RWMutex
	Updaters   []chan<- anyvec.Vector
	UpdatersWg sync.WaitGroup

	// Lock for reading during all calls; lock for
	// writing during an actual Close.
	CloseLock sync.RWMutex
	Closed    bool
}

// VanillaParamServer creates a ParamServer that applies
// vanilla gradient updates.
//
// The params argument specifies the subset of the agent's
// parameters to update.
// The anynet.AllParameters function must be able to see
// all of the parameters.
func VanillaParamServer(agent *Agent, params []*anydiff.Var,
	stepSize float64) ParamServer {
	return newParamServer(agent, params, stepSize, func() anysgd.Transformer {
		return nil
	})
}

// RMSPropParamServer creates a ParamServer that applies
// shared RMSProp.
//
// The arguments are similar to the arguments for
// VanillaParamServer.
func RMSPropParamServer(agent *Agent, params []*anydiff.Var,
	stepSize float64, r anysgd.RMSProp) ParamServer {
	return newParamServer(agent, params, stepSize, func() anysgd.Transformer {
		return &anysgd.RMSProp{
			DecayRate: r.DecayRate,
			Damping:   r.Damping,
		}
	})
}

func newParamServer(agent *Agent, params []*anydiff.Var, stepSize float64,
	trans func() anysgd.Transformer) *paramServer {
	res := &paramServer{
		StepSize: stepSize,
		Agent:    agent,

		Params:   params,
		Locks:    make([]*sync.RWMutex, len(params)),
		Updaters: make([]chan<- anyvec.Vector, len(params)),
	}
	for i, param := range params {
		ch := make(chan anyvec.Vector, 1)
		lock := &sync.RWMutex{}
		res.Locks[i] = lock
		res.Updaters[i] = ch
		tr := trans()
		res.UpdatersWg.Add(1)
		if tr != nil {
			go transformerUpdater(param, ch, lock, &res.UpdatersWg, stepSize, tr)
		} else {
			go vanillaUpdater(param, ch, lock, &res.UpdatersWg, stepSize)
		}
	}
	return res
}

func (p *paramServer) LocalCopy() (agent *LocalAgent, err error) {
	defer essentials.AddCtxTo("copy global agent", &err)

	p.CloseLock.RLock()
	defer p.CloseLock.RUnlock()
	if p.Closed {
		return nil, errClosed
	}

	for _, lock := range p.Locks {
		lock.RLock()
		defer lock.RUnlock()
	}

	copied, err := p.Agent.Copy()
	if err != nil {
		return nil, err
	}

	globalToLocal := map[*anydiff.Var]*anydiff.Var{}
	locals := anynet.AllParameters(copied.Base, copied.Actor, copied.Critic)
	globals := anynet.AllParameters(p.Agent.Base, p.Agent.Actor, p.Agent.Critic)
	for i, globalParam := range globals {
		globalToLocal[globalParam] = locals[i]
	}

	var params []*anydiff.Var
	for i, globalParam := range p.Params {
		local, ok := globalToLocal[globalParam]
		if !ok {
			return nil, fmt.Errorf("parameter %d not found via anynet.AllParameters", i)
		}
		params = append(params, local)
	}

	return &LocalAgent{
		Agent:  copied,
		Params: params,
	}, nil
}

func (p *paramServer) Sync(l *LocalAgent) (err error) {
	defer essentials.AddCtxTo("sync local agent", &err)

	p.CloseLock.RLock()
	defer p.CloseLock.RUnlock()
	if p.Closed {
		return errClosed
	}

	var wg sync.WaitGroup
	for i, localParam := range l.Params {
		wg.Add(1)
		go func(i int, localParam *anydiff.Var) {
			p.Locks[i].RLock()
			localParam.Vector.Set(p.Params[i].Vector)
			p.Locks[i].RUnlock()
			wg.Done()
		}(i, localParam)
	}

	wg.Wait()
	return nil
}

func (p *paramServer) Update(g anydiff.Grad, l *LocalAgent) (err error) {
	defer essentials.AddCtxTo("sync local agent", &err)

	p.CloseLock.RLock()
	defer p.CloseLock.RUnlock()
	if p.Closed {
		return errClosed
	}

	var wg sync.WaitGroup
	for i, localParam := range l.Params {
		wg.Add(1)
		go func(i int, localParam *anydiff.Var) {
			p.Updaters[i] <- g[localParam]
			wg.Done()
		}(i, localParam)
	}

	// Even though the update is asynchronous, we block in
	// order to allow for back-pressure.
	wg.Wait()

	return nil
}

func (p *paramServer) Close() error {
	p.CloseLock.Lock()
	defer p.CloseLock.Unlock()
	if !p.Closed {
		p.Closed = true
		for _, ch := range p.Updaters {
			close(ch)
		}
		p.UpdatersWg.Wait()
	}
	return nil
}

func vanillaUpdater(param *anydiff.Var, ch <-chan anyvec.Vector,
	lock *sync.RWMutex, wg *sync.WaitGroup, stepSize float64) {
	defer wg.Done()
	for change := range ch {
		lock.Lock()
		change.Scale(change.Creator().MakeNumeric(stepSize))
		param.Vector.Add(change)
		lock.Unlock()
	}
}

func transformerUpdater(param *anydiff.Var, ch <-chan anyvec.Vector,
	lock *sync.RWMutex, wg *sync.WaitGroup, stepSize float64,
	trans anysgd.Transformer) {
	defer wg.Done()
	for change := range ch {
		grad := anydiff.Grad{param: change}
		grad = trans.Transform(grad)
		lock.Lock()
		vec := grad[param]
		vec.Scale(vec.Creator().MakeNumeric(stepSize))
		param.Vector.Add(vec)
		lock.Unlock()
	}
}
