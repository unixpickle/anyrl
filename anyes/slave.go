package anyes

import (
	"time"

	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

// StopConds is a set of stopping conditions for an
// environment rollout.
//
// If a field is set to its 0 value, that stopping
// condition is not used.
type StopConds struct {
	// MaxTime is the maximum time for which a rollout
	// should be run.
	MaxTime time.Duration

	// MaxSteps is the maximum number of steps to take
	// in the environment.
	MaxSteps int
}

// Rollout contains information about the result of
// running an agent on an environment.
type Rollout struct {
	// Scale used to generate the rollout.
	Scale float64

	// Seed used to generate the rollout.
	Seed int64

	// Reward is the cumulative reward.
	Reward float64

	// Steps is the number of steps taken.
	Steps int

	// EarlyStop is true if the rollout ended because
	// of a stop condition rather than because of a
	// terminal state.
	EarlyStop bool
}

// A Slave is a slave node from a master's point of view.
//
// A Slave is not assumed to be thread-safe.
// Only one Slave method can be running at once.
type Slave interface {
	// Init tells the slave about the initial model
	// parameters and the pre-generated random noise.
	Init(modelData []byte, noiseSeed int64, noiseSize int) error

	// Run runs the environment with the given seed
	// and returns the resulting reward.
	//
	// The randomized mutation vector should be scaled
	// by the given scaler before being added.
	Run(stop *StopConds, scale float64, seed int64) (*Rollout, error)

	// Update computes a new set of parameters by
	// adding a certain amount of noise for each of
	// the given seeds.
	// It returns a Checksum of the new parameters.
	Update(scales []float64, seeds []int64) (Checksum, error)
}

// AnynetSlave is a Slave which works by running an RNN
// block on a pre-determined environment.
type AnynetSlave struct {
	Creator anyvec.Creator
	Params  *AnynetParams
	Policy  anyrnn.Block
	Env     anyrl.Env

	// Sampler, if non-nil, is applied to Policy outputs
	// right before they are fed to the environment.
	Sampler anyrl.Sampler

	// NoiseGroup is used to generate noise.
	// If it is nil, Init sets it.
	NoiseGroup *NoiseGroup
}

// Init updates the parameters and initializes the noise
// generator.
func (a *AnynetSlave) Init(data []byte, seed int64, size int) (err error) {
	defer essentials.AddCtxTo("init AnynetSlave", &err)

	err = a.Params.SetData(data)
	if err != nil {
		return
	}

	if a.NoiseGroup == nil {
		a.NoiseGroup = &NoiseGroup{}
	}
	return a.NoiseGroup.Init(seed, size)
}

// Run executes an environment rollout.
func (a *AnynetSlave) Run(stop *StopConds, scale float64, seed int64) (r *Rollout,
	err error) {
	defer essentials.AddCtxTo("run AnynetSlave", &err)

	oldData, err := a.Params.Data()
	if err != nil {
		return
	}
	defer func() {
		if subErr := a.Params.SetData(oldData); err == nil {
			err = subErr
		}
	}()
	mutation := a.NoiseGroup.Gen(scale, seed, a.Params.Len())
	a.Params.SplitMutation(mutation).AddToVars()

	r = &Rollout{
		Scale:     scale,
		Seed:      seed,
		EarlyStop: true,
	}
	obs, err := a.Env.Reset()
	if err != nil {
		return
	}
	state := a.Policy.Start(1)
	var timeout <-chan time.Time
	if stop.MaxTime != 0 {
		timeout = time.After(stop.MaxTime)
	}
	for r.Steps < stop.MaxSteps || stop.MaxSteps == 0 {
		select {
		case <-timeout:
			return
		default:
		}

		out := a.Policy.Step(state, anyvec.Make(a.Creator, obs))
		state = out.State()
		action := out.Output()
		if a.Sampler != nil {
			action = a.Sampler.Sample(action, 1)
		}

		var rew float64
		var done bool
		obs, rew, done, err = a.Env.Step(a.Creator.Float64Slice(action.Data()))
		if err != nil {
			return
		}
		r.Reward += rew
		r.Steps++
		if done {
			r.EarlyStop = false
			return
		}
	}
	return
}

// Update updates the parameters by re-generating the
// mutations and adding them.
func (a *AnynetSlave) Update(scales []float64, seeds []int64) (Checksum, error) {
	vec := a.NoiseGroup.GenSum(scales, seeds, a.Params.Len())
	a.Params.Update(vec)
	return a.Params.Checksum()
}
