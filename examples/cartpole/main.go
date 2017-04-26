package main

import (
	"log"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec/anyvec32"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
)

const (
	Host             = "localhost:5001"
	RolloutsPerBatch = 30
	NumBatches       = 50

	// Set to true if you want to watch the AI learn.
	// Makes everything very slow.
	RenderEnv = false

	// Set to true if you plan to upload the monitor
	// to the website.
	CaptureVideo = false
)

func main() {
	// Connect to gym server.
	client, err := gym.Make(Host, "CartPole-v0")
	must(err)
	defer client.Close()

	// Start monitoring.
	monitorFile := "gym-monitor"
	must(client.Monitor(monitorFile, true, false, CaptureVideo))

	// Create a neural network policy.
	creator := anyvec32.CurrentCreator()
	policy := &anyrnn.LayerBlock{
		Layer: anynet.Net{
			anynet.NewFC(creator, 4, 32),
			anynet.Tanh,
			anynet.NewFC(creator, 32, 16),
			anynet.Tanh,
			anynet.NewFC(creator, 16, 2),
		},
	}
	actionSampler := anyrl.Softmax{}

	// Create an anyrl.Env from our gym environment.
	env, err := anyrl.GymEnv(creator, client, RenderEnv)
	must(err)

	// Setup Trust Region Policy Optimization for training.
	trpo := &anyrl.TRPO{
		NaturalPG: anyrl.NaturalPG{
			Policy:      policy,
			Params:      policy.Parameters(),
			ActionSpace: actionSampler,
			Damping:     0.1,
		},
	}

	for batchIdx := 0; batchIdx < NumBatches; batchIdx++ {
		// Gather episode rollouts.
		var rollouts []*anyrl.RolloutSet
		for i := 0; i < RolloutsPerBatch; i++ {
			rollout, err := anyrl.RolloutRNN(creator, policy, actionSampler, env)
			must(err)
			rollouts = append(rollouts, rollout)
		}

		// Join the rollouts into one set.
		r := anyrl.PackRolloutSets(rollouts)

		// Print the rewards.
		log.Printf("batch %d: mean_reward=%v", batchIdx, r.MeanReward(creator))

		// Train on the rollouts.
		grad := trpo.Run(r)
		grad.AddToVars()
	}

	// Uncomment to upload to OpenAI Gym.
	// You will have to set OPENAI_GYM_API_KEY.
	//
	//     client.Close()
	//     must(gym.Upload(Host, monitorFile, "", ""))
	//
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
