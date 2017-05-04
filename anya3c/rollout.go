package anya3c

import (
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
)

// rollout represents a (partial) trajectory through an
// environment.
type rollout struct {
	Outs    [][]anyrnn.Res
	Rewards []float64
	Sampled []anyvec.Vector
}

// runRollout generates a rollout by sampling actions
// using a worker.
//
// If maxSteps is non-zero, it limits the number of
// timesteps in the environment.
func runRollout(w *worker, maxSteps int) (*rollout, error) {
	var r rollout
	for t := 0; t < maxSteps || maxSteps == 0; t++ {
		w.StepAgent()
		lastOut := w.AgentRes
		reward, action, err := w.StepEnv()
		if err != nil {
			return nil, err
		}
		r.add(lastOut, reward, action)
		if w.EnvDone {
			break
		}
	}
	return &r, nil
}

// Advantages computes an empirical advantage function
// estimator at every timestep.
//
// Since this may be bootstrapped, the worker may be used
// to run the critic on the next observation.
func (r *rollout) Advantages(w *worker, discount float64) []anyvec.Numeric {
	c := w.Agent.Params[0].Vector.Creator()
	ops := c.NumOps()

	followingReward := c.MakeNumeric(0)
	if !w.EnvDone {
		// Bootstrap from value function.
		followingReward = anyvec.Sum(w.PeekCritic())
	}
	discountNum := c.MakeNumeric(discount)

	advantages := make([]anyvec.Numeric, len(r.Rewards))
	for t := len(r.Rewards) - 1; t >= 0; t-- {
		followingReward = ops.Add(c.MakeNumeric(r.Rewards[t]),
			ops.Mul(discountNum, followingReward))
		criticOut := anyvec.Sum(r.Outs[t][2].Output())
		advantages[t] = ops.Sub(followingReward, criticOut)
	}

	return advantages
}

// add appends a frame of experience.
func (r *rollout) add(out []anyrnn.Res, reward float64, sampled anyvec.Vector) {
	r.Outs = append(r.Outs, out)
	r.Rewards = append(r.Rewards, reward)
	r.Sampled = append(r.Sampled, sampled)
}
