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
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

const (
	Host       = "localhost:5001"
	NumWorkers = 4
)

const (
	RenderEnv = false

	NetworkSaveFile = "trained_agent"
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

	// Create a neural network agent.
	agent := loadOrCreateAgent(creator)

	paramServer := anya3c.RMSPropParamServer(agent, agent.AllParameters(),
		1e-4, anysgd.RMSProp{DecayRate: 0.99})

	a3c := &anya3c.A3C{
		ParamServer: paramServer,
		Logger: &anya3c.AvgLogger{
			Creator: creator,
			Logger: &anya3c.StandardLogger{
				Episode:    true,
				Update:     true,
				Regularize: true,
			},
			// Only log updates and entropy periodically.
			Update:     400,
			Regularize: 8000,
		},
		Discount: 0.99,
		MaxSteps: 20,
		Regularizer: &anypg.EntropyReg{
			Entropyer: anyrl.Softmax{},
			Coeff:     0.003,
		},
	}

	log.Println("Press Ctrl+C to stop.")
	err := a3c.Run(envs, rip.NewRIP().Chan())
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// Necessary to safely save the agent.
	paramServer.Close()

	must(serializer.SaveAny(NetworkSaveFile, agent.Base, agent.Actor, agent.Critic))
}

func loadOrCreateAgent(creator anyvec.Creator) *anya3c.Agent {
	var res anya3c.Agent
	res.ActionSpace = anyrl.Softmax{}
	err := serializer.LoadAny(NetworkSaveFile, &res.Base, &res.Actor, &res.Critic)
	if err == nil {
		log.Println("Loaded network from file.")
		return &res
	} else {
		log.Println("Created new network.")
		res.Base = &anyrnn.LayerBlock{
			Layer: anynet.Net{
				// Most inputs are 0, so we can amplify the effect
				// of non-zero inputs a bit.
				anynet.NewAffine(creator, 8, 0),

				// Fully-connected MLP.
				anynet.NewFC(creator, PreprocessedSize, 96),
				anynet.Tanh,
				anynet.NewFC(creator, 96, 64),
				anynet.Tanh,
			},
		}
		res.Actor = &anyrnn.LayerBlock{
			Layer: anynet.Net{
				// Zero initialization encourages random exploration.
				anynet.NewFCZero(creator, 64, 6),
			},
		}
		res.Critic = &anyrnn.LayerBlock{
			Layer: anynet.Net{
				// Don't zero initialize; it messes up RMSProp when
				// the first reward is finally seen.
				anynet.NewFC(creator, 64, 1),
			},
		}
		return &res
	}
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
