package anyrl

import (
	"errors"
	"fmt"

	"github.com/unixpickle/essentials"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
)

var errSpaceLength = errors.New("space vector has incorrect length")

// Env is an instance of an RL environment.
//
// Both observations and actions are fixed-size vectors.
type Env interface {
	Reset() (observation []float64, err error)
	Step(action []float64) (observation []float64,
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
func GymEnv(e gym.Env, render bool) (env Env, err error) {
	defer essentials.AddCtxTo("create gym Env", &err)
	actionSpace, err := e.ActionSpace()
	if err != nil {
		return nil, err
	}
	obsSpace, err := e.ObservationSpace()
	if err != nil {
		return nil, err
	}
	actConv, err := converterForSpace(actionSpace)
	if err != nil {
		return nil, err
	}
	obsConv, err := converterForSpace(obsSpace)
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

func (g *gymEnv) Reset() (obsVec []float64, err error) {
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

func (g *gymEnv) Step(action []float64) (obsVec []float64, reward float64,
	done bool, err error) {
	defer essentials.AddCtxTo("step gym Env", &err)
	gymAction, err := g.actConv.ToGym(action)
	if err != nil {
		return
	}
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
	VecLen() int
	ToGym(in []float64) (interface{}, error)
	FromGym(in gym.Obs) ([]float64, error)
}

func converterForSpace(s *gym.Space) (gymSpaceConverter, error) {
	switch s.Type {
	case "Box":
		vecLen := 1
		for _, x := range s.Shape {
			vecLen *= x
		}
		return &boxSpaceConverter{Len: vecLen}, nil
	case "Discrete":
		return &discreteSpaceConverter{N: s.N}, nil
	case "MultiBinary":
		return &multiBinarySpaceConverter{N: s.N}, nil
	case "Tuple":
		var subConvs []gymSpaceConverter
		for _, subSpace := range s.Subspaces {
			subConv, err := converterForSpace(subSpace)
			if err != nil {
				return nil, err
			}
			subConvs = append(subConvs, subConv)
		}
		return &tupleSpaceConverter{Spaces: subConvs}, nil
	default:
		return nil, errors.New("unsupported space: " + s.Type)
	}
}

type boxSpaceConverter struct {
	Len int
}

func (b *boxSpaceConverter) VecLen() int {
	return b.Len
}

func (b *boxSpaceConverter) ToGym(in []float64) (interface{}, error) {
	if len(in) != b.VecLen() {
		return nil, errSpaceLength
	}
	return in, nil
}

func (b *boxSpaceConverter) FromGym(in gym.Obs) ([]float64, error) {
	return gym.Flatten(in)
}

type discreteSpaceConverter struct {
	N int
}

func (d *discreteSpaceConverter) VecLen() int {
	return d.N
}

func (d *discreteSpaceConverter) ToGym(in []float64) (interface{}, error) {
	if len(in) != d.VecLen() {
		return nil, errSpaceLength
	}
	for i, x := range in {
		if x != 0 {
			return i, nil
		}
	}
	return nil, errors.New("no one-hot value is set")
}

func (d *discreteSpaceConverter) FromGym(in gym.Obs) ([]float64, error) {
	var num int
	if err := in.Unmarshal(&num); err != nil {
		return nil, err
	}
	out := make([]float64, d.N)
	out[num] = 1
	return out, nil
}

type multiBinarySpaceConverter struct {
	N int
}

func (m *multiBinarySpaceConverter) VecLen() int {
	return m.N
}

func (m *multiBinarySpaceConverter) ToGym(in []float64) (interface{}, error) {
	if len(in) != m.VecLen() {
		return nil, errSpaceLength
	}
	for _, x := range in {
		if x != 0 && x != 1 {
			return nil, fmt.Errorf("unexpected multi-binary value: %v", x)
		}
	}
	return in, nil
}

func (m *multiBinarySpaceConverter) FromGym(in gym.Obs) ([]float64, error) {
	var vec []float64
	if err := in.Unmarshal(&vec); err != nil {
		return nil, err
	}
	return vec, nil
}

type tupleSpaceConverter struct {
	Spaces []gymSpaceConverter
}

func (t *tupleSpaceConverter) VecLen() int {
	var size int
	for _, s := range t.Spaces {
		size += s.VecLen()
	}
	return size
}

func (t *tupleSpaceConverter) ToGym(in []float64) (interface{}, error) {
	if len(in) != t.VecLen() {
		return nil, errSpaceLength
	}
	var res []interface{}
	for _, s := range t.Spaces {
		subVec := in[:s.VecLen()]
		in = in[s.VecLen():]
		gymObj, err := s.ToGym(subVec)
		if err != nil {
			return nil, err
		}
		res = append(res, gymObj)
	}
	return res, nil
}

func (t *tupleSpaceConverter) FromGym(in gym.Obs) ([]float64, error) {
	subObs, err := gym.UnpackTuple(in)
	if err != nil {
		return nil, err
	}
	if len(subObs) != len(t.Spaces) {
		return nil, fmt.Errorf("expected %d tuple elements but got %d", len(t.Spaces),
			len(subObs))
	}
	var reses []float64
	for i, obs := range subObs {
		subVec, err := t.Spaces[i].FromGym(obs)
		if err != nil {
			return nil, err
		}
		reses = append(reses, subVec...)
	}
	return reses, nil
}
