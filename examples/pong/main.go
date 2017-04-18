package main

import (
	"compress/flate"
	"log"
	"sync"

	gym "github.com/openai/gym-http-api/binding-go"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/lazyseq/lazyrnn"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

// It is recommended that you launch multiple instances
// of the HTTP server for optimal performance.
//
// In Bash, you can do this via:
//
//     $ for i in 500{0..3}; do python gym_http_server.py -p $i & done
//
// Then, you can kill the tasks via:
//
//     $ jobs -p | xargs kill
//
var BaseURLs = []string{
	"http://localhost:5000",
	"http://localhost:5001",
	"http://localhost:5002",
	"http://localhost:5003",
}

var BatchSize = 32 / len(BaseURLs)

const (
	RenderEnv = true

	FrameWidth  = 160
	FrameHeight = 210

	NetworkSaveFile = "trained_policy"
)

func main() {
	// Setup vector creator.
	creator := anyvec32.CurrentCreator()

	// Create multiple environment instances so that we
	// can record multiple episodes at once.
	var envs []anyrl.Env
	for _, baseURL := range BaseURLs {
		// Connect to gym server.
		client, err := gym.NewClient(baseURL)
		must(err)

		id, err := client.Create("Pong-v0")
		must(err)
		defer client.Close(id)

		// Create an anyrl.Env from our gym environment.
		env, err := anyrl.GymEnv(creator, client, id, RenderEnv)
		must(err)

		envs = append(envs, env)
	}

	// Create a neural network policy.
	policy := loadOrCreateNetwork(creator)
	actionSampler := anyrl.Softmax{}

	// Setup Trust Region Policy Optimization for training.
	trpo := &anyrl.TRPO{
		NaturalPG: anyrl.NaturalPG{
			Policy:      policy,
			Params:      policy.Parameters(),
			ActionSpace: actionSampler,

			// Apply the RNN in a low-memory way.
			ApplyPolicy: func(seq lazyseq.Rereader, b anyrnn.Block) lazyseq.Rereader {
				out := lazyrnn.FixedHSM(1, true, seq, b)

				// Stores the RNN outputs in memory, but nothing else.
				return lazyseq.Lazify(lazyseq.Unlazify(out))
			},
		},
	}

	// Train on a background goroutine so that we can
	// listen for Ctrl+C on the main goroutine.
	var trainLock sync.Mutex
	go func() {
		for batchIdx := 0; true; batchIdx++ {
			log.Println("Gathering batch of experience...")

			// Gather episode rollouts.
			var rollouts []*anyrl.RolloutSet
			for i := 0; i < BatchSize; i++ {
				rollout := rollout(creator, policy, envs)
				log.Printf("batch %d: sub_batch=%d mean_reward=%v", batchIdx, i,
					rollout.MeanReward(creator))
				rollouts = append(rollouts, rollout)
			}

			// Join the rollouts into one set.
			r := anyrl.PackRolloutSets(rollouts)

			// Print the rewards.
			log.Printf("batch %d: mean_reward=%v", batchIdx, r.MeanReward(creator))

			// Train on the rollouts.
			log.Println("Training on batch...")
			trainLock.Lock()
			grad := trpo.Run(r)
			grad.AddToVars()
			trainLock.Unlock()
		}
	}()

	log.Println("Running. Press Ctrl+C to stop.")
	<-rip.NewRIP().Chan()

	// Avoid the race condition where we save during
	// training.
	trainLock.Lock()
	must(serializer.SaveAny(NetworkSaveFile, policy))
}

func loadOrCreateNetwork(creator anyvec.Creator) anyrnn.Stack {
	var res anyrnn.Stack
	if err := serializer.LoadAny(NetworkSaveFile, &res); err == nil {
		log.Println("Loaded network from file.")
		return res
	} else {
		log.Println("Created new network.")
		inputSize := FrameWidth * FrameHeight * 3
		return anyrnn.Stack{
			// Use a vanilla RNN so we can remember what we saw
			// in the previous frame.
			anyrnn.NewVanilla(creator, inputSize, 64, anynet.Tanh),
			&anyrnn.LayerBlock{
				Layer: anynet.Net{
					anynet.NewFC(creator, 64, 32),
					anynet.Tanh,
					anynet.NewFCZero(creator, 32, 6),
				},
			},
		}
	}
}

func rollout(creator anyvec.Creator, agent anyrnn.Block, envs []anyrl.Env) *anyrl.RolloutSet {
	// Compress the input frames as we store them.
	// If we used a ReferenceTape for the input, the
	// program would use way too much memory.
	inputs, inputCh := lazyseq.CompressedTape(flate.DefaultCompression)

	rewards, rewardsCh := lazyseq.ReferenceTape()
	sampled, sampledCh := lazyseq.ReferenceTape()

	err := anyrl.RolloutRNNChans(creator, agent, anyrl.Softmax{},
		inputCh, rewardsCh, sampledCh, envs...)
	must(err)

	close(inputCh)
	close(rewardsCh)
	close(sampledCh)

	return &anyrl.RolloutSet{Inputs: inputs, Rewards: rewards, SampledOuts: sampled}
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
