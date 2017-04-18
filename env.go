package anyrl

import (
	"errors"
	"fmt"

	gym "github.com/openai/gym-http-api/binding-go"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

// Env is an instance of an RL environment.
type Env interface {
	Reset() (observation anyvec.Vector, err error)
	Step(action anyvec.Vector) (observation anyvec.Vector,
		reward float64, done bool, err error)
}

type gymEnv struct {
	client *gym.Client
	id     gym.InstanceID
	render bool

	actConv gymSpaceConverter
	obsConv gymSpaceConverter
}

// GymEnv creates an Env from an OpenAI Gym instance.
//
// This will fail if the instance requires an unsupported
// space type or if it fails to fetch space info.
func GymEnv(c anyvec.Creator, client *gym.Client,
	id gym.InstanceID, render bool) (env Env, err error) {
	defer essentials.AddCtxTo("create gym Env", &err)
	actionSpace, err := client.ActionSpace(id)
	if err != nil {
		return nil, err
	}
	obsSpace, err := client.ObservationSpace(id)
	if err != nil {
		return nil, err
	}
	actConv, err := converterForSpace(c, actionSpace)
	if err != nil {
		return nil, err
	}
	obsConv, err := converterForSpace(c, obsSpace)
	if err != nil {
		return nil, err
	}
	return &gymEnv{
		client:  client,
		id:      id,
		actConv: actConv,
		obsConv: obsConv,
		render:  render,
	}, nil
}

func (g *gymEnv) Reset() (obsVec anyvec.Vector, err error) {
	defer essentials.AddCtxTo("reset gym Env", &err)
	obs, err := g.client.Reset(g.id)
	if err != nil {
		return nil, err
	}
	return g.obsConv.FromGym(obs)
}

func (g *gymEnv) Step(action anyvec.Vector) (obsVec anyvec.Vector, reward float64,
	done bool, err error) {
	defer essentials.AddCtxTo("step gym Env", &err)
	gymAction := g.actConv.ToGym(action)
	var obs interface{}
	obs, reward, done, _, err = g.client.Step(g.id, gymAction, g.render)
	if err != nil {
		return
	}
	obsVec, err = g.obsConv.FromGym(obs)
	return
}

type gymSpaceConverter interface {
	ToGym(in anyvec.Vector) interface{}
	FromGym(in interface{}) (anyvec.Vector, error)
}

func converterForSpace(c anyvec.Creator, s *gym.Space) (gymSpaceConverter, error) {
	switch s.Name {
	case "Box":
		return &boxSpaceConverter{Creator: c}, nil
	case "Discrete":
		return &discreteSpaceConverter{Creator: c, N: s.N}, nil
	default:
		return nil, errors.New("unsupported space: " + s.Name)
	}
}

type boxSpaceConverter struct {
	Creator anyvec.Creator
}

func (b *boxSpaceConverter) ToGym(in anyvec.Vector) interface{} {
	switch data := in.Data().(type) {
	case []float64:
		return data
	case []float32:
		res := make([]float64, len(data))
		for i, x := range data {
			res[i] = float64(x)
		}
		return res
	default:
		panic(fmt.Sprintf("unsupported numeric list: %T", data))
	}
}

func (b *boxSpaceConverter) FromGym(in interface{}) (anyvec.Vector, error) {
	switch in := in.(type) {
	case []float64:
		return b.Creator.MakeVectorData(b.Creator.MakeNumericList(in)), nil
	case [][]float64:
		var joined []float64
		for _, x := range in {
			joined = append(joined, x...)
		}
		return b.FromGym(joined)
	case [][][]float64:
		var joined [][]float64
		for _, x := range in {
			joined = append(joined, x...)
		}
		return b.FromGym(joined)
	default:
		return nil, fmt.Errorf("unexpected observation type: %T", in)
	}
}

type discreteSpaceConverter struct {
	Creator anyvec.Creator
	N       int
}

func (d *discreteSpaceConverter) ToGym(in anyvec.Vector) interface{} {
	return anyvec.MaxIndex(in)
}

func (d *discreteSpaceConverter) FromGym(in interface{}) (anyvec.Vector, error) {
	var inInt int
	switch in := in.(type) {
	case int:
		inInt = in
	case float64:
		inInt = int(in)
	default:
		return nil, fmt.Errorf("unexpected observation type: %T", in)
	}
	if inInt < 0 || inInt >= d.N {
		return nil, fmt.Errorf("discrete observation out of bounds: %d", inInt)
	}
	out := make([]float64, d.N)
	out[inInt] = 1
	return d.Creator.MakeVectorData(d.Creator.MakeNumericList(out)), nil
}
