package main

import (
	"log"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec/anyvec32"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/rip"
)

const (
	Host      = "localhost:5001"
	BatchSize = 20

	// Set to true if you want to watch the AI learn.
	// Makes everything very slow.
	RenderEnv = false

	// Set to true if you plan to upload the monitor
	// to the website.
	CaptureVideo = false
)

func main() {
	// Create the gym environment.
	client, err := gym.Make(Host, "Copy-v0")
	must(err)
	defer client.Close()

	// Start monitoring.
	monitorFile := "gym-monitor"
	must(client.Monitor(monitorFile, true, false, CaptureVideo))

	// Setup vector creator.
	creator := anyvec32.CurrentCreator()

	// Wrap the gym environment in an anyrl.Env.
	env, err := anyrl.GymEnv(client, RenderEnv)
	must(err)

	// Create a neural network policy.
	policy := &anyrnn.LayerBlock{
		Layer: anynet.Net{
			anynet.NewFC(creator, 6, 32),
			anynet.Tanh,

			// Zero last layer encourages exploration.
			anynet.NewFCZero(creator, 32, 2+5),
		},
	}

	// Actions are drawn from the tuple space described by
	// Tuple(Discrete(2), Discrete(2), Discrete(5)).
	actionSpace := &anyrl.Tuple{
		Spaces: []interface{}{
			&anyrl.Bernoulli{OneHot: true},
			&anyrl.Bernoulli{OneHot: true},
			anyrl.Softmax{},
		},
		ParamSizes:  []int{1, 1, 5},
		SampleSizes: []int{2, 2, 5},
	}

	// Setup Trust Region Policy Optimization for training.
	trpo := &anypg.TRPO{
		NaturalPG: anypg.NaturalPG{
			Policy:      policy,
			Params:      policy.Parameters(),
			ActionSpace: actionSpace,

			// Optimization for feed-forward policy.
			ApplyPolicy: func(seq lazyseq.Rereader, b anyrnn.Block) lazyseq.Rereader {
				return lazyseq.Map(seq, b.(*anyrnn.LayerBlock).Layer.Apply)
			},

			// Focus on short-term rewards.
			ActionJudger: &anypg.QJudger{Discount: 0.9},
		},
	}

	// Setup an RNNRoller for producing rollouts.
	roller := &anyrl.RNNRoller{Block: policy, ActionSpace: actionSpace}

	log.Println("Press Ctrl+C to finish training.")
	r := rip.NewRIP()
	var batchIdx int
	for !r.Done() {
		var rollouts []*anyrl.RolloutSet
		for i := 0; i < BatchSize; i++ {
			rollout, err := roller.Rollout(env)
			must(err)
			rollouts = append(rollouts, rollout)
		}

		// Join the rollouts into one set.
		packed := anyrl.PackRolloutSets(creator, rollouts)

		// Print stats for the batch.
		entropyReg := &anypg.EntropyReg{Entropyer: actionSpace, Coeff: 1}
		entropy := anypg.AverageReg(packed.AgentOuts, entropyReg)
		log.Printf("batch %d: mean=%f entropy=%f", batchIdx, packed.Rewards.Mean(),
			entropy)
		batchIdx++

		// Train on the rollouts.
		grad := trpo.Run(packed)
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
