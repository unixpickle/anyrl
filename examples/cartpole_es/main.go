package main

import (
	"log"
	"runtime"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anyes"
	"github.com/unixpickle/anyvec/anyvec32"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
	"github.com/unixpickle/serializer"
)

const (
	Host             = "localhost:5001"
	RolloutsPerBatch = 50
	NumBatches       = 50

	// Set to true if you want to watch the AI learn.
	// Makes everything very slow.
	RenderEnv = false
)

func main() {
	// Create a neural network policy.
	creator := anyvec32.CurrentCreator()
	policy := &anyrnn.LayerBlock{
		Layer: anynet.Net{
			anynet.NewFC(creator, 4, 32),
			anynet.Tanh,
			anynet.NewFC(creator, 32, 16),
			anynet.Tanh,

			// Encourage exploration with a zero last layer.
			anynet.NewFCZero(creator, 16, 1),
		},
	}
	actionSampler := &anyrl.Bernoulli{OneHot: true}

	// Setup the main coordinator for Evolution Strategies.
	master := &anyes.Master{
		Noise: anyes.NewNoise(1337, 1<<15),
		Params: anyes.MakeSafe(&anyes.AnynetParams{
			Params: anynet.AllParameters(policy),
		}),
		Normalize:   true,
		NoiseStddev: 0.1,
		StepSize:    0.1,
	}

	// Setup slaves for Evolution Strategies.
	group := &anyes.NoiseGroup{}
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		// Connect to gym server.
		client, err := gym.Make(Host, "CartPole-v0")
		must(err)
		defer client.Close()

		// Create an anyrl.Env from our gym environment.
		env, err := anyrl.GymEnv(creator, client, RenderEnv)
		must(err)

		// Create a copy of the model.
		newPolicy, err := serializer.Copy(policy)
		must(err)
		slave := &anyes.AnynetSlave{
			Params: &anyes.AnynetParams{
				Params: anynet.AllParameters(newPolicy),
			},
			Policy:     newPolicy.(anyrnn.Block),
			Env:        env,
			Sampler:    actionSampler,
			NoiseGroup: group,
		}
		must(master.AddSlave(slave))
	}

	for batchIdx := 0; batchIdx < NumBatches; batchIdx++ {
		// Get a batch of mutations and rewards.
		stopConds := &anyes.StopConds{MaxSteps: 200}
		r, err := master.Rollouts(stopConds, RolloutsPerBatch/2)
		must(err)

		// Print the rewards.
		log.Printf("batch %d: mean_reward=%f", batchIdx, anyes.MeanReward(r))

		// Train on the rollouts.
		must(master.Update(r))
	}
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
