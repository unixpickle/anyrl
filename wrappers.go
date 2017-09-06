package anyrl

import (
	"errors"

	"github.com/unixpickle/anyvec"
)

// MetaEnv is a meta-learning environment in which
// episodes consist of one or more episodes of a contained
// environment.
//
// The action space is unchanged, but the observations are
// augmented (at the end) with the previous action, the
// reward, and the done value (in that order).
//
// For the first observation, the action, reward, and done
// values are set to 0.
type MetaEnv struct {
	Env

	// NumRuns is the number of times to run Env in each
	// meta-episode.
	NumRuns int

	// ActionSize is the size of action vectors.
	// It is used by Reset() to create a zero last-action
	// vector.
	ActionSize int

	runsRemaining int
}

// Reset resets the environment.
func (m *MetaEnv) Reset() (obs anyvec.Vector, err error) {
	m.runsRemaining = m.NumRuns
	obs, err = m.Env.Reset()
	if err != nil {
		return
	}
	zeroVec := obs.Creator().MakeVector(m.ActionSize + 2)
	obs = obs.Creator().Concat(obs, zeroVec)
	return
}

// Step takes a step in the environment.
func (m *MetaEnv) Step(act anyvec.Vector) (obs anyvec.Vector, rew float64,
	done bool, err error) {
	if m.runsRemaining <= 0 {
		err = errors.New("step: done sub-episodes in meta-environment")
		return
	}
	obs, rew, done, err = m.Env.Step(act)
	if err != nil {
		return
	}
	rewDoneVec := []float64{rew, 0}
	if done {
		rewDoneVec[1] = 1
		m.runsRemaining--
		done = m.runsRemaining == 0
		if !done {
			obs, err = m.Env.Reset()
			if err != nil {
				return
			}
		}
	}
	c := obs.Creator()
	obs = c.Concat(obs, act, c.MakeVectorData(c.MakeNumericList(rewDoneVec)))
	return
}

// MaxStepsEnv wraps an Env and ends episodes early if
// they run longer than MaxSteps timesteps.
type MaxStepsEnv struct {
	Env
	MaxSteps int

	steps int
}

// Reset resets the environment.
func (m *MaxStepsEnv) Reset() (anyvec.Vector, error) {
	m.steps = 0
	return m.Env.Reset()
}

// Step takes a step in the environment.
func (m *MaxStepsEnv) Step(action anyvec.Vector) (anyvec.Vector, float64, bool, error) {
	obs, rew, done, err := m.Env.Step(action)
	m.steps++
	if m.steps == m.MaxSteps {
		done = true
	}
	return obs, rew, done, err
}
