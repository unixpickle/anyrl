// Package anya3c implements Asynchronous Advantage
// Actor-Critic, a Reinforcement Learning algorithm from
// https://arxiv.org/abs/1602.01783.
package anya3c

import (
	"sync"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/essentials"
)

// A3C holds the configuration for an instance of A3C.
type A3C struct {
	ParamServer ParamServer
	Logger      Logger

	// Discount is the reward discount factor.
	//
	// If 0, then no discount is used.
	Discount float64

	// MaxSteps is the maximum number of steps to take before
	// doing a parameter update.
	//
	// If 0, then episodes are completed before updates.
	MaxSteps int

	// Regularizer is used to regularize the actor.
	//
	// If nil, no regularization is used.
	Regularizer anypg.Regularizer
}

// Run runs A3C with a worker for each environment.
//
// If the done channel is closed, this finishes gracefully
// and returns nil.
// If any environment produces an error, this stops and
// returns the error.
func (a *A3C) Run(envs []anyrl.Env, done <-chan struct{}) (err error) {
	defer essentials.AddCtxTo("run A3C", &err)

	errChan := make(chan error, len(envs))
	stopChan := make(chan struct{})

	var wg sync.WaitGroup
	for i, e := range envs {
		wg.Add(1)
		go func(i int, e anyrl.Env) {
			defer wg.Done()
			if err := a.worker(i, e, stopChan); err != nil {
				errChan <- err
			}
		}(i, e)
	}

	select {
	case err = <-errChan:
	case <-done:
	}
	close(stopChan)

	wg.Wait()
	return
}

func (a *A3C) worker(id int, env anyrl.Env, stopChan <-chan struct{}) error {
	w, err := newWorker(id, env, a.ParamServer)
	if err != nil {
		return err
	}
	if err := w.Reset(); err != nil {
		return err
	}

	for {
		select {
		case <-stopChan:
			return nil
		default:
		}

		if err := a.update(w); err != nil {
			return err
		}
		if w.EnvDone {
			if a.Logger != nil {
				a.Logger.LogEpisode(w.ID, w.RewardSum)
			}
			if err := w.Reset(); err != nil {
				return err
			}
		}
	}
}

func (a *A3C) update(w *worker) error {
	if err := a.ParamServer.Sync(w.Agent); err != nil {
		return err
	}
	r, err := runRollout(w, a.MaxSteps)
	if err != nil {
		return err
	}
	bptt := &bptt{
		Rollout:     r,
		Worker:      w,
		Discount:    a.Discount,
		Regularizer: a.Regularizer,
		Logger:      a.Logger,
	}
	if bptt.Discount == 0 {
		bptt.Discount = 1
	}
	grad, mse := bptt.Run()
	if err := a.ParamServer.Update(grad, w.Agent); err != nil {
		return err
	}
	if a.Logger != nil {
		a.Logger.LogUpdate(w.ID, mse)
	}
	return nil
}
