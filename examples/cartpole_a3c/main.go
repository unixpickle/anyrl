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
	policyBase := &anyrnn.LayerBlock{
		Layer: anynet.Net{
			anynet.NewFC(creator, 4, 64),
			anynet.Tanh,
		},
	}
	policyActor := &anyrnn.LayerBlock{
		Layer: anynet.NewFCZero(creator, 64, 1),
	}
	policyCritic := &anyrnn.LayerBlock{
		Layer: anynet.Net{
			anynet.NewFC(creator, 64, 32),
			anynet.ReLU,
			anynet.NewFCZero(creator, 32, 1),
		},
	}
	actionSpace := &anyrl.Bernoulli{OneHot: true}

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
	a3c := &anya3c.A3C{
		PolicyBase:   policyBase,
		PolicyActor:  policyActor,
		PolicyCritic: policyCritic,

		ActionSpace: actionSpace,
		Params:      anynet.AllParameters(policyBase, policyActor, policyCritic),
		StepSize:    StepSize,
		MaxSteps:    5,
		Discount:    0.9,
		Transformer: &anysgd.RMSProp{DecayRate: 0.99},

		Log: func(str string) {
			log.Println(str)
		},
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
