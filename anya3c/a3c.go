// Package anya3c implements Asynchronous Advantage
// Actor-Critic for Reinforcement Learning.
package anya3c

import (
	"fmt"
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

type A3CActionSpace interface {
	anyrl.Sampler
	anyrl.LogProber
}

// A3C implements asynchronous advantage actor-critic.
type A3C struct {
	// The policy is split up into three parts.
	//
	// Each input is fed into PolicyBase.
	// The output of PolicyBase is fed into PolicyActor
	// and PolicyCritic, which implement the actor and
	// the critic respectively.
	//
	// These blocks must work with serializer.Copy.
	PolicyBase   anyrnn.Block
	PolicyActor  anyrnn.Block
	PolicyCritic anyrnn.Block

	ActionSpace A3CActionSpace
	Params      []*anydiff.Var
	StepSize    float64

	// Log, if non-nil, is used to log information about
	// training as it happens.
	Log func(str string)

	// Discount is the reward discount factor.
	//
	// If 0, 1 is used.
	Discount float64

	// MaxSteps is the maximum episode length between
	// updates.
	//
	// If 0, infinity is used.
	MaxSteps int

	// Transformer is the global gradient transformer.
	// It is synchronized and used from all threads.
	//
	// If nil, vanilla gradients are used.
	Transformer anysgd.Transformer

	// Regularizer is used to regularize the actor's
	// action space.
	//
	// If nil, no regularization is used.
	Regularizer anypg.Regularizer
}

// Run runs A3C with a different actor thread for each
// environment.
//
// If the done channel is closed, this finishes gracefully
// and returns nil.
//
// If any environment produces an error, this stops and
// returns the error.
func (a *A3C) Run(envs []anyrl.Env, done <-chan struct{}) (err error) {
	defer essentials.AddCtxTo("run A3C", &err)

	errChan := make(chan error, len(envs))
	stopChan := make(chan struct{})
	globals := &a3cGlobals{
		DoneChan: stopChan,
	}

	var wg sync.WaitGroup
	for i, e := range envs {
		wg.Add(1)
		go func(i int, e anyrl.Env) {
			defer wg.Done()
			if err := a.worker(i, e, globals); err != nil {
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

func (a *A3C) worker(id int, e anyrl.Env, g *a3cGlobals) error {
	locals, err := newA3CLocals(e, a, g)
	if err != nil {
		return err
	}
	for !g.Done() {
		if err := a.workerStep(locals, g); err != nil {
			return err
		}
		if locals.EnvDone {
			a.log("worker %d: reward=%f", id, locals.RewardSum)
			if err := locals.Reset(a, g); err != nil {
				return err
			}
		}
	}
	return nil
}

func (a *A3C) workerStep(l *a3cLocals, g *a3cGlobals) error {
	var rollout a3cRollout
	for t := 0; t < a.MaxSteps || a.MaxSteps == 0; t++ {
		l.StepPolicy()
		lastOut := l.PolicyRes
		reward, action, err := l.StepEnv(a.ActionSpace)
		if err != nil {
			return err
		}
		rollout.Add(lastOut, reward, action)
		if l.EnvDone {
			break
		}
	}

	grad := a.bptt(l, &rollout)

	globalGrad := anydiff.Grad{}
	for param, vec := range grad {
		globalGrad[l.LocalToGlobal[param]] = vec
	}

	g.TransformerLock.Lock()
	if a.Transformer != nil {
		globalGrad = a.Transformer.Transform(globalGrad)
	}
	g.ParamLock.Lock()
	globalGrad.Scale(l.Creator.MakeNumeric(a.StepSize))
	globalGrad.AddToVars()
	g.ParamLock.Unlock()
	g.TransformerLock.Unlock()

	l.UpdateParams(a, g)

	return nil
}

func (a *A3C) bptt(l *a3cLocals, r *a3cRollout) anydiff.Grad {
	var gradVars []*anydiff.Var
	for _, param := range a.Params {
		gradVars = append(gradVars, l.GlobalToLocal[param])
	}
	grad := anydiff.NewGrad(gradVars...)

	c := l.Creator
	ops := c.NumOps()

	followingReward := c.MakeNumeric(0)
	if !l.EnvDone {
		// Bootstrap from value function.
		l.StepPolicy()
		followingReward = anyvec.Sum(l.PolicyRes[2].Output())
	}
	discount := c.MakeNumeric(a.discount())

	stateUpstream := make([]anyrnn.StateGrad, 3)
	for t := len(r.Rewards) - 1; t >= 0; t-- {
		outReses := r.Outs[t]
		followingReward = ops.Add(c.MakeNumeric(r.Rewards[t]),
			ops.Mul(discount, followingReward))
		criticOut := anyvec.Sum(outReses[2].Output())
		advantage := ops.Sub(followingReward, criticOut)

		criticUpstream := c.MakeVector(1)
		criticUpstream.AddScalar(ops.Mul(advantage, c.MakeNumeric(2)))
		actorUpstream := a.actorUpstream(outReses[1].Output(), r.Sampled[t],
			advantage)

		var baseUpstream1, baseUpstream2 anyvec.Vector
		baseUpstream1, stateUpstream[1] = outReses[1].Propagate(actorUpstream,
			stateUpstream[1], grad)
		baseUpstream2, stateUpstream[2] = outReses[2].Propagate(criticUpstream,
			stateUpstream[2], grad)

		baseUpstream1.Add(baseUpstream2)

		_, stateUpstream[0] = outReses[0].Propagate(baseUpstream1,
			stateUpstream[0], grad)
	}

	return grad
}

func (a *A3C) actorUpstream(params, choice anyvec.Vector,
	advantage anyvec.Numeric) anyvec.Vector {
	c := params.Creator()
	v := anydiff.NewVar(params)
	g := anydiff.NewGrad(v)
	upstream := c.MakeVector(1)
	upstream.AddScalar(advantage)
	a.ActionSpace.LogProb(v, choice, 1).Propagate(upstream, g)

	if a.Regularizer != nil {
		penalty := a.Regularizer.Regularize(v, 1)
		upstream.SetData(c.MakeNumericList([]float64{1}))
		penalty.Propagate(upstream, g)
	}

	return g[v]
}

func (a *A3C) discount() float64 {
	if a.Discount == 0 {
		return 1
	} else {
		return a.Discount
	}
}

func (a *A3C) log(format string, args ...interface{}) {
	if a.Log != nil {
		a.Log(fmt.Sprintf(format, args...))
	}
}

type a3cGlobals struct {
	DoneChan <-chan struct{}

	TransformerLock sync.Mutex
	ParamLock       sync.RWMutex
}

func (a *a3cGlobals) Done() bool {
	select {
	case <-a.DoneChan:
		return true
	default:
		return false
	}
}

// a3cLocals stores and manipulates the state of a worker.
type a3cLocals struct {
	Creator anyvec.Creator

	// Order: base, actor, critic.
	PolicyBlocks []anyrnn.Block
	PolicyStates []anyrnn.State
	PolicyRes    []anyrnn.Res

	LocalToGlobal map[*anydiff.Var]*anydiff.Var
	GlobalToLocal map[*anydiff.Var]*anydiff.Var

	Env       anyrl.Env
	EnvState  anyvec.Vector
	EnvDone   bool
	RewardSum float64
}

func newA3CLocals(e anyrl.Env, a *A3C, g *a3cGlobals) (*a3cLocals, error) {
	startState, err := e.Reset()
	if err != nil {
		return nil, err
	}
	res := &a3cLocals{
		Creator:       startState.Creator(),
		PolicyBlocks:  []anyrnn.Block{a.PolicyBase, a.PolicyActor, a.PolicyCritic},
		LocalToGlobal: map[*anydiff.Var]*anydiff.Var{},
		GlobalToLocal: map[*anydiff.Var]*anydiff.Var{},
		Env:           e,
		EnvState:      startState,
	}

	g.ParamLock.RLock()
	defer g.ParamLock.RUnlock()
	for i, block := range res.PolicyBlocks {
		newBlock, err := serializer.Copy(block)
		if err != nil {
			return nil, err
		}
		b := newBlock.(anyrnn.Block)
		res.PolicyBlocks[i] = b
		res.PolicyStates = append(res.PolicyStates, b.Start(1))
	}

	oldParams := anynet.AllParameters(a.PolicyBase, a.PolicyActor, a.PolicyCritic)
	newParams := anynet.AllParameters(res.PolicyBlocks[0], res.PolicyBlocks[1],
		res.PolicyBlocks[2])
	for i, old := range oldParams {
		res.GlobalToLocal[old] = newParams[i]
		res.LocalToGlobal[newParams[i]] = old
	}

	return res, nil
}

// Reset resets the environment and the RNN state.
func (a *a3cLocals) Reset(a3c *A3C, g *a3cGlobals) error {
	startState, err := a.Env.Reset()
	if err != nil {
		return err
	}
	a.EnvState = startState
	a.EnvDone = false
	for i, block := range a.PolicyBlocks {
		a.PolicyStates[i] = block.Start(1)
	}
	a.PolicyRes = nil
	a.RewardSum = 0
	return nil
}

// UpdateParams copies the global parameters into the
// local copy of the policy.
func (a *a3cLocals) UpdateParams(a3c *A3C, g *a3cGlobals) {
	g.ParamLock.RLock()
	defer g.ParamLock.RUnlock()
	for old, new := range a.GlobalToLocal {
		new.Vector.Set(old.Vector)
	}
}

// StepPolicy evaluates the policy on the current
// environment state and sets a.PolicyRes.
// If a.PolicyRes is already set, this does nothing.
func (a *a3cLocals) StepPolicy() {
	if a.PolicyRes != nil {
		return
	}
	a.PolicyRes = []anyrnn.Res{
		a.PolicyBlocks[0].Step(a.PolicyStates[0], a.EnvState),
	}
	baseOut := a.PolicyRes[0].Output()
	for i := 1; i < 3; i++ {
		out := a.PolicyBlocks[i].Step(a.PolicyStates[i], baseOut)
		a.PolicyRes = append(a.PolicyRes, out)
		a.PolicyStates[i] = out.State()
	}
}

// StepEnv uses the policy output from a.PolicyRes to
// take a step in the environment.
// It resets a.PolicyRes to nil.
func (a *a3cLocals) StepEnv(actionSpace A3CActionSpace) (reward float64,
	action anyvec.Vector, err error) {
	action = actionSpace.Sample(a.PolicyRes[1].Output(), 1)
	a.EnvState, reward, a.EnvDone, err = a.Env.Step(action)
	if err != nil {
		return
	}
	a.RewardSum += reward
	a.PolicyRes = nil
	return
}

type a3cRollout struct {
	Outs    [][]anyrnn.Res
	Rewards []float64
	Sampled []anyvec.Vector
}

func (r *a3cRollout) Add(out []anyrnn.Res, reward float64, sampled anyvec.Vector) {
	r.Outs = append(r.Outs, out)
	r.Rewards = append(r.Rewards, reward)
	r.Sampled = append(r.Sampled, sampled)
}
