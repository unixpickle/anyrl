package anyrl

import (
	"errors"
	"fmt"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
)

// Env is an instance of an RL environment.
type Env interface {
	Reset() (observation anyvec.Vector, err error)
	Step(action anyvec.Vector) (observation anyvec.Vector,
		reward float64, done bool, err error)
}

type gymEnv struct {
	env    gym.Env
	render bool

	actConv gymSpaceConverter
	obsConv gymSpaceConverter
}

// GymEnv creates an Env from an OpenAI Gym instance.
//
// This will fail if the instance requires an unsupported
// space type or if it fails to fetch space info.
//
// If render is true, then the environment will be
// graphically rendered after every step.
func GymEnv(c anyvec.Creator, e gym.Env, render bool) (env Env, err error) {
	defer essentials.AddCtxTo("create gym Env", &err)
	actionSpace, err := e.ActionSpace()
	if err != nil {
		return nil, err
	}
	obsSpace, err := e.ObservationSpace()
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
		env:     e,
		actConv: actConv,
		obsConv: obsConv,
		render:  render,
	}, nil
}

func (g *gymEnv) Reset() (obsVec anyvec.Vector, err error) {
	defer essentials.AddCtxTo("reset gym Env", &err)
	obs, err := g.env.Reset()
	if err != nil {
		return nil, err
	}
	if g.render {
		if err := g.env.Render(); err != nil {
			return nil, err
		}
	}
	return g.obsConv.FromGym(obs)
}

func (g *gymEnv) Step(action anyvec.Vector) (obsVec anyvec.Vector, reward float64,
	done bool, err error) {
	defer essentials.AddCtxTo("step gym Env", &err)
	gymAction := g.actConv.ToGym(action)
	var obs gym.Obs
	obs, reward, done, _, err = g.env.Step(gymAction)
	if err != nil {
		return
	}
	if g.render {
		if err = g.env.Render(); err != nil {
			return
		}
	}
	obsVec, err = g.obsConv.FromGym(obs)
	return
}

type gymSpaceConverter interface {
	ToGym(in anyvec.Vector) interface{}
	FromGym(in gym.Obs) (anyvec.Vector, error)
}

func converterForSpace(c anyvec.Creator, s *gym.Space) (gymSpaceConverter, error) {
	switch s.Type {
	case "Box":
		return &boxSpaceConverter{Creator: c}, nil
	case "Discrete":
		return &discreteSpaceConverter{Creator: c, N: s.N}, nil
	default:
		return nil, errors.New("unsupported space: " + s.Type)
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

func (b *boxSpaceConverter) FromGym(in gym.Obs) (anyvec.Vector, error) {
	vec, err := gym.Flatten(in)
	if err != nil {
		return nil, err
	}
	return b.Creator.MakeVectorData(b.Creator.MakeNumericList(vec)), nil
}

type discreteSpaceConverter struct {
	Creator anyvec.Creator
	N       int
}

func (d *discreteSpaceConverter) ToGym(in anyvec.Vector) interface{} {
	return anyvec.MaxIndex(in)
}

func (d *discreteSpaceConverter) FromGym(in gym.Obs) (anyvec.Vector, error) {
	var num int
	if err := in.Unmarshal(&num); err != nil {
		return nil, err
	}
	out := make([]float64, d.N)
	out[num] = 1
	return d.Creator.MakeVectorData(d.Creator.MakeNumericList(out)), nil
}
