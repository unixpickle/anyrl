package main

import (
	"compress/flate"
	"log"
	"sync"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

const (
	Host         = "localhost:5001"
	ParallelEnvs = 8
	BatchSize    = 32 / ParallelEnvs
)

const (
	RenderEnv = false

	NetworkSaveFile = "trained_policy"
)

func main() {
	// Setup vector creator.
	creator := anyvec32.CurrentCreator()

	// Create multiple environment instances so that we
	// can record multiple episodes at once.
	var envs []anyrl.Env
	for i := 0; i < ParallelEnvs; i++ {
		// Connect to gym server.
		client, err := gym.Make(Host, "Pong-v0")
		must(err)

		defer client.Close()

		// Create an anyrl.Env from our gym environment.
		env, err := anyrl.GymEnv(creator, client, RenderEnv)
		must(err)

		envs = append(envs, &PreprocessEnv{Env: env})
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

			ApplyPolicy: func(seq lazyseq.Rereader, b anyrnn.Block) lazyseq.Rereader {
				// Utilize the fact that the model is feed-forward.
				out := lazyseq.Map(seq, b.(*anyrnn.LayerBlock).Layer.Apply)

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

func loadOrCreateNetwork(creator anyvec.Creator) *anyrnn.LayerBlock {
	var res *anyrnn.LayerBlock
	if err := serializer.LoadAny(NetworkSaveFile, &res); err == nil {
		log.Println("Loaded network from file.")
		return res
	} else {
		log.Println("Created new network.")
		return &anyrnn.LayerBlock{
			Layer: anynet.Net{
				// Most inputs are 0, so we can amplify the effect
				// of non-zero inputs a bit.
				anynet.NewAffine(creator, 8, 0),

				// Fully-connected network with 256 hidden units.
				anynet.NewFC(creator, PreprocessedSize, 256),
				anynet.Tanh,
				anynet.NewFCZero(creator, 256, 6),
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
