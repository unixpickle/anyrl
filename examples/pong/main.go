package main

import (
	"compress/flate"
	"log"
	"sync"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anymisc"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/lazyseq/lazyrnn"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

const (
	Host         = "localhost:5001"
	ParallelEnvs = 8
	BatchSteps   = 100000
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
	trpo := &anypg.TRPO{
		NaturalPG: anypg.NaturalPG{
			Policy:      policy,
			Params:      policy.Parameters(),
			ActionSpace: actionSampler,

			// Speed things up a bit.
			Iters: 4,

			ApplyPolicy: func(seq lazyseq.Rereader, b anyrnn.Block) lazyseq.Rereader {
				out := lazyrnn.FixedHSM(30, false, seq, b)
				return lazyseq.Lazify(lazyseq.Unlazify(out))
			},

			Regularizer: &anypg.EntropyReg{
				Entropyer: actionSampler,
				Coeff:     0.01,
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
			var steps int
			for steps < BatchSteps {
				rollout := rollout(creator, policy, envs)
				steps += rollout.NumSteps()
				log.Printf("batch %d: steps=%d sub_mean=%v", batchIdx, steps,
					rollout.MeanReward(creator))
				rollouts = append(rollouts, rollout)
			}

			// Join the rollouts into one set.
			r := anyrl.PackRolloutSets(rollouts)

			// Print the stats for the batch.
			ops := creator.NumOps()
			log.Printf("batch %d: mean=%v stddev=%v", batchIdx,
				r.MeanReward(creator),
				ops.Pow(r.RewardVariance(creator), creator.MakeNumeric(0.5)))

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
		return anyrnn.Stack{
			&anyrnn.LayerBlock{
				Layer: anynet.Net{
					// Most inputs are 0, so we can amplify the effect
					// of non-zero inputs a bit.
					anynet.NewAffine(creator, 8, 0),
				},
			},
			anymisc.NewNPRNN(creator, PreprocessedSize, 256),
			&anyrnn.LayerBlock{
				Layer: anynet.NewFCZero(creator, 256, 6),
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
