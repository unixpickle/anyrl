package anyes

import "time"

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
	Update(scales []float64, seeds []int64) error
}
