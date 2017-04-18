package main

import (
	"log"
	"os"
	"path/filepath"

	gym "github.com/openai/gym-http-api/binding-go"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec/anyvec32"
)

const (
	BaseURL          = "http://localhost:5000"
	RolloutsPerBatch = 5
	NumBatches       = 50
	RenderEnv        = false
)

func main() {
	// Connect to gym server.
	client, err := gym.NewClient(BaseURL)
	must(err)

	// Create environment instance.
	id, err := client.Create("CartPole-v0")
	must(err)
	defer client.Close(id)

	// Start monitoring to "./cartpole-monitor".
	workingDir, err := os.Getwd()
	must(err)
	monitorFile := filepath.Join(workingDir, "gym-monitor")
	must(client.StartMonitor(id, monitorFile, false, false, false))
	defer client.CloseMonitor(id)

	// Create a neural network policy.
	creator := anyvec32.CurrentCreator()
	policy := &anyrnn.LayerBlock{
		Layer: anynet.Net{
			anynet.NewFC(creator, 4, 8),
			anynet.Tanh,
			anynet.NewFC(creator, 8, 2),
		},
	}
	actionSampler := anyrl.Softmax{}

	// Create an anyrl.Env from our gym environment.
	env, err := anyrl.GymEnv(creator, client, id, RenderEnv)
	must(err)

	// Setup Trust Region Policy Optimization for training.
	trpo := &anyrl.TRPO{
		NaturalPG: anyrl.NaturalPG{
			Policy:      policy,
			Params:      policy.Parameters(),
			ActionSpace: actionSampler,
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
		log.Printf("batch %d: mean=%v rewards=%v", batchIdx, r.MeanReward(creator),
			r.TotalRewards(creator).Data())

		// Train on the rollouts.
		grad := trpo.Run(r)
		grad.AddToVars()
	}

	// Uncomment the code below to upload to the Gym website.
	// Note: you must set the OPENAI_GYM_API_KEY environment
	// variable or set the second argument of Upload() to a
	// non-empty string.
	//
	//     must(client.Upload("cartpole-monitor", "", ""))
	//
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
