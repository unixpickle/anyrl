package main

import (
	"log"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec/anyvec32"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
	"github.com/unixpickle/lazyseq"
)

const (
	Host             = "localhost:5001"
	RolloutsPerBatch = 30
	BatchEpochs      = 5
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

	// Create a neural network agent.
	creator := anyvec32.CurrentCreator()
	policy := anynet.Net{
		anynet.NewFC(creator, 4, 32),
		anynet.Tanh,
		anynet.NewFC(creator, 32, 16),
		anynet.Tanh,

		// Encourage exploration with a zero last layer.
		anynet.NewFCZero(creator, 16, 1),
	}
	critic := anynet.Net{
		anynet.NewFC(creator, 4, 32),
		anynet.Tanh,
		anynet.NewFC(creator, 32, 1),
	}
	actionSampler := &anyrl.Bernoulli{OneHot: true}

	// Create an anyrl.Env from our gym environment.
	env, err := anyrl.GymEnv(creator, client, RenderEnv)
	must(err)

	// Setup Trust Region Policy Optimization for training.
	ppo := &anypg.PPO{
		Params: anynet.AllParameters(policy, critic),
		Actor: func(in lazyseq.Rereader) lazyseq.Rereader {
			return lazyseq.Map(in, policy.Apply)
		},
		Critic: func(in lazyseq.Rereader) lazyseq.Rereader {
			return lazyseq.Map(in, critic.Apply)
		},
		ActionSpace: actionSampler,
		Discount:    0.99,
		Lambda:      1,
	}

	// Setup an RNNRoller to collect episode rollouts.
	roller := &anyrl.RNNRoller{
		Block:       &anyrnn.LayerBlock{Layer: policy},
		ActionSpace: actionSampler,
	}

	var transformer anysgd.Adam
	stepSize := 0.01
	for batchIdx := 0; batchIdx < NumBatches; batchIdx++ {
		// Gather episode rollouts.
		var rollouts []*anyrl.RolloutSet
		for i := 0; i < RolloutsPerBatch; i++ {
			rollout, err := roller.Rollout(env)
			must(err)
			rollouts = append(rollouts, rollout)
		}

		// Join the rollouts into one set.
		r := anyrl.PackRolloutSets(rollouts)

		// Print the rewards.
		log.Printf("batch %d: mean_reward=%f", batchIdx, r.Rewards.Mean())

		// Train on the rollouts.
		for i := 0; i < BatchEpochs; i++ {
			grad := ppo.Run(r)
			g := transformer.Transform(grad)
			g.Scale(creator.MakeNumeric(stepSize))
			g.AddToVars()
		}
		stepSize *= 0.9
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
