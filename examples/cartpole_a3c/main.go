package main

import (
	"log"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anya3c"
	"github.com/unixpickle/anyvec/anyvec32"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
	"github.com/unixpickle/rip"
)

const (
	Host     = "localhost:5001"
	Workers  = 4
	StepSize = 1e-3

	// Set to true to watch a worker.
	Render = false
)

func main() {
	// Construct our policy + critic.
	creator := anyvec32.CurrentCreator()
	agent := &anya3c.Agent{
		Base: &anyrnn.LayerBlock{
			Layer: anynet.Net{
				anynet.NewFC(creator, 4, 64),
				anynet.Tanh,
			},
		},
		Actor: &anyrnn.LayerBlock{
			Layer: anynet.NewFCZero(creator, 64, 1),
		},
		Critic: &anyrnn.LayerBlock{
			Layer: anynet.Net{
				anynet.NewFC(creator, 64, 32),
				anynet.ReLU,
				anynet.NewFCZero(creator, 32, 1),
			},
		},
		ActionSpace: &anyrl.Bernoulli{OneHot: true},
	}

	// Create environments.
	envs := make([]anyrl.Env, Workers)
	for i := range envs {
		// Connect to gym server.
		client, err := gym.Make(Host, "CartPole-v0")
		must(err)
		defer client.Close()

		envs[i], err = anyrl.GymEnv(creator, client, Render && i == 0)
		must(err)
	}

	// Train via A3C, stopping on Ctrl+C.
	paramServer := anya3c.RMSPropParamServer(agent, agent.AllParameters(),
		StepSize, anysgd.RMSProp{DecayRate: 0.99})
	defer paramServer.Close()
	a3c := &anya3c.A3C{
		ParamServer: paramServer,

		// Log out reward averages and critic MSE averages.
		Logger: &anya3c.AvgLogger{
			Creator: creator,
			Logger: &anya3c.StandardLogger{
				Episode: true,
				Update:  true,
			},
			Episode: 30,
			Update:  3000,
		},

		MaxSteps: 5,
		Discount: 0.9,
	}

	log.Println("Press Ctrl+C to stop learning.")
	err := a3c.Run(envs, rip.NewRIP().Chan())

	must(err)
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
