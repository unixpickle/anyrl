package anyrl

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestRNNRoller(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	block := anyrnn.NewLSTM(c, 3, 4)
	actionSpace := Softmax{}
	roller := &RNNRoller{
		Block:       block,
		ActionSpace: actionSpace,
	}
	envs := make([]Env, 5)
	seqLens := make([]int, len(envs))

	for i := range envs {
		randObs := c.MakeVector(3)
		anyvec.Rand(randObs, anyvec.Normal, nil)
		seqLens[i] = 1 + rand.Intn(20)
		envs[i] = &rnnTestEnv{
			RewardScale: rand.Float64(),
			EpLen:       seqLens[i],
			Observation: randObs,
		}
	}

	rollouts, err := roller.Rollout(envs...)
	if err != nil {
		t.Fatal(err)
	}

	actualSeqLens := make([]int, len(envs))
	state := block.Start(len(envs))
	var timestep int
	for observations := range rollouts.Inputs.ReadTape(0, -1) {
		state = state.Reduce(observations.Present)

		actionsBatch := <-rollouts.Actions.ReadTape(timestep, timestep+1)
		if !reflect.DeepEqual(actionsBatch.Present, observations.Present) {
			t.Errorf("time %d: actions present should be %v but got %v",
				timestep, actionsBatch.Present, observations.Present)
		}
		actions := actionsBatch.Packed
		for i, p := range observations.Present {
			if p {
				actualSeqLens[i]++
				actualReward := rollouts.Rewards[i][timestep]
				expectedReward := envs[i].(*rnnTestEnv).RewardScale *
					float64(anyvec.MaxIndex(actions.Slice(0, 4)))
				actions = actions.Slice(4, actions.Len())
				if math.Abs(actualReward-expectedReward) > 1e-4 {
					t.Errorf("time %d: seq %d: expected reward %f but got %f",
						timestep, i, expectedReward, actualReward)
				}
			}
		}

		res := block.Step(state, observations.Packed)
		state = res.State()

		agentOutBatch := <-rollouts.AgentOuts.ReadTape(timestep, timestep+1)
		if !reflect.DeepEqual(agentOutBatch.Present, observations.Present) {
			t.Errorf("time %d: agent out present should be %v but got %v",
				timestep, agentOutBatch.Present, observations.Present)
		}

		expectedOut := res.Output()
		outDiff := expectedOut.Copy()
		outDiff.Sub(agentOutBatch.Packed)
		if anyvec.AbsMax(outDiff).(float64) > 1e-4 {
			t.Errorf("time %d: output should be %v but got %v", timestep,
				expectedOut.Data(), agentOutBatch.Packed.Data())
		}

		timestep++
	}

	if !reflect.DeepEqual(seqLens, actualSeqLens) {
		t.Errorf("expected seq lens %v but got %v", seqLens, actualSeqLens)
	}
}

// rnnTestEnv is a deterministic environment with
// controllable behavior, making it ideal for testing
// rollouts.
type rnnTestEnv struct {
	RewardScale float64
	EpLen       int
	Observation anyvec.Vector

	timestep int
}

func (r *rnnTestEnv) Reset() (anyvec.Vector, error) {
	r.timestep = 1
	return r.obsVec(), nil
}

func (r *rnnTestEnv) Step(action anyvec.Vector) (obs anyvec.Vector, rew float64,
	done bool, err error) {
	obs = r.obsVec()
	rew = float64(anyvec.MaxIndex(action)) * r.RewardScale
	done = r.timestep == r.EpLen
	r.timestep++
	return
}

func (r *rnnTestEnv) obsVec() anyvec.Vector {
	res := r.Observation.Copy()
	res.Scale(res.Creator().MakeNumeric(float64(r.timestep)))
	return res
}
