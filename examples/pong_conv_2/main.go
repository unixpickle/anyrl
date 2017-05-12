package main

import (
	"compress/flate"
	"log"
	"math"
	"sync"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
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
	log.Println("Creating environments...")
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
	actionSpace := anyrl.Softmax{}

	// Setup an RNNRoller for producing rollouts.
	roller := &anyrl.RNNRoller{
		Block:       policy,
		ActionSpace: actionSpace,

		// Compress the input frames as we store them.
		// If we used a ReferenceTape for the input, the
		// program would use way too much memory.
		MakeInputTape: func() (lazyseq.Tape, chan<- *anyseq.Batch) {
			return lazyseq.CompressedUint8Tape(flate.DefaultCompression)
		},
	}

	// Setup Trust Region Policy Optimization for training.
	trpo := &anypg.TRPO{
		NaturalPG: anypg.NaturalPG{
			Policy:      policy,
			Params:      policy.Parameters(),
			ActionSpace: actionSpace,

			// Speed things up a bit.
			Iters: 10,
			Reduce: (&anyrl.FracReducer{
				Frac:          0.1,
				MakeInputTape: roller.MakeInputTape,
			}).Reduce,

			ApplyPolicy: func(seq lazyseq.Rereader, b anyrnn.Block) lazyseq.Rereader {
				out := lazyrnn.FixedHSM(30, false, seq, b)
				return lazyseq.Lazify(lazyseq.Unlazify(out))
			},
			ActionJudger: &anypg.QJudger{Discount: 0.99},
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
				rollout, err := roller.Rollout(envs...)
				must(err)
				steps += rollout.NumSteps()
				log.Printf("batch %d: steps=%d sub_mean=%f", batchIdx, steps,
					rollout.Rewards.Mean())
				rollouts = append(rollouts, rollout)
			}

			// Join the rollouts into one set.
			r := anyrl.PackRolloutSets(rollouts)

			// Print the stats for the batch.
			log.Printf("batch %d: mean=%f stddev=%f", batchIdx,
				r.Rewards.Mean(), math.Sqrt(r.Rewards.Variance()))

			// Train on the rollouts.
			log.Println("Training on batch...")
			grad := trpo.Run(r)
			trainLock.Lock()
			for i, param := range anynet.AllParameters(policy) {
				log.Println("param", i, "mag", anyvec.Norm(grad[param]))
			}
			grad.AddToVars()
			trainLock.Unlock()
		}
	}()

	log.Println("Running. Press Ctrl+C to stop.")
	<-rip.NewRIP().Chan()

	// Avoid the race condition where we save during
	// parameter updates.
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
		markup := `
			Input(w=80, h=105, d=3)

			Linear(scale=0.003921568627, bias=-0.5647058824)
			#Residual {
			#	Projection {
			#		Linear(scale=-1)
			#		ReLU
			#	}
			#	ReLU
			#}

			Linear(scale=24)

			Conv(w=4, h=4, n=16, sx=2, sy=2)
			ReLU
			Conv(w=4, h=4, n=32, sx=2, sy=2)
			ReLU
			FC(out=256)
			ReLU
			#Conv(w=4, h=4, n=16, sx=2, sy=2)
			#ReLU
			#FC(out=128)
			#ReLU
		`
		convNet, err := anyconv.FromMarkup(creator, markup)
		must(err)
		net := convNet.(anynet.Net)
		net = setupVisionLayers(net)
		return anyrnn.Stack{
			&anyrnn.LayerBlock{Layer: net},
			anymisc.NewNPRNN(creator, 256, 256),
			&anyrnn.LayerBlock{
				Layer: anynet.NewFCZero(creator, 256, 6),
			},
		}
	}
}

func setupVisionLayers(net anynet.Net) anynet.Net {
	for _, layer := range net {
		//projectOutSolidColors(layer)
		boostBiases(layer)
	}
	return net
}

func projectOutSolidColors(layer anynet.Layer) {
	switch layer := layer.(type) {
	case *anyconv.Conv:
		filters := layer.Filters.Vector
		inDepth := layer.InputDepth
		numFilters := layer.FilterCount
		filterSize := filters.Len() / numFilters
		for i := 0; i < numFilters; i++ {
			filter := filters.Slice(i*filterSize, (i+1)*filterSize)

			// Compute the mean for each input channel.
			negMean := anyvec.SumRows(filter, inDepth)
			negMean.Scale(negMean.Creator().MakeNumeric(-1 / float64(filterSize/inDepth)))
			anyvec.AddRepeated(filter, negMean)
		}
	case *anynet.FC:
		negMean := anyvec.SumCols(layer.Weights.Vector, layer.OutCount)
		negMean.Scale(negMean.Creator().MakeNumeric(-1 / float64(layer.InCount)))
		anyvec.AddChunks(layer.Weights.Vector, negMean)
	}
}

func boostBiases(layer anynet.Layer) {
	switch layer := layer.(type) {
	case *anyconv.Conv:
		bias := layer.Biases.Vector.Creator().MakeNumeric(1)
		layer.Biases.Vector.AddScalar(bias)
	case *anynet.FC:
		bias := layer.Biases.Vector.Creator().MakeNumeric(1)
		layer.Biases.Vector.AddScalar(bias)
	}
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
