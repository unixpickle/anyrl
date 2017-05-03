package main

import (
	"fmt"
	"log"
	"os"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anya3c"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

const (
	Host       = "localhost:5001"
	NumWorkers = 8
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
	for i := 0; i < NumWorkers; i++ {
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
	base, actor, critic := loadOrCreateNetwork(creator)
	actionSampler := anyrl.Softmax{}

	a3c := &anya3c.A3C{
		PolicyBase:   base,
		PolicyActor:  actor,
		PolicyCritic: critic,
		ActionSpace:  actionSampler,
		Params:       anynet.AllParameters(base, actor, critic),
		StepSize:     0.001,
		Log: func(str string) {
			log.Println(str)
		},
		Discount:    0.99,
		MaxSteps:    5,
		Transformer: &anysgd.RMSProp{DecayRate: 0.99},
		Entropyer:   actionSampler,
		EntropyReg:  0.01,
	}

	log.Println("Press Ctrl+C to stop.")
	err := a3c.Run(envs, rip.NewRIP().Chan())
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	must(serializer.SaveAny(NetworkSaveFile, base, actor, critic))
}

func loadOrCreateNetwork(creator anyvec.Creator) (base, actor, critic *anyrnn.LayerBlock) {
	if err := serializer.LoadAny(NetworkSaveFile, &base, &actor, &critic); err == nil {
		log.Println("Loaded network from file.")
		return
	} else {
		log.Println("Created new network.")
		base = &anyrnn.LayerBlock{
			Layer: anynet.Net{
				// Most inputs are 0, so we can amplify the effect
				// of non-zero inputs a bit.
				anynet.NewAffine(creator, 8, 0),

				// Fully-connected network with 256 hidden units.
				anynet.NewFC(creator, PreprocessedSize, 256),
				anynet.ReLU,
			},
		}
		actor = &anyrnn.LayerBlock{
			Layer: anynet.Net{
				anynet.NewFCZero(creator, 256, 6),
			},
		}
		critic = &anyrnn.LayerBlock{
			Layer: anynet.Net{
				anynet.NewFCZero(creator, 256, 1),
			},
		}
		return
	}
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
