package anyes

import (
	"crypto/md5"
	"errors"
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

// ParamVersion is a version number used by SafeParams.
type ParamVersion int64

// Checksum is a set of bits identifying some parameters.
// If two sets of parameters are the same, their Checksums
// should match.
// If they are different, there should be a low chance
// that the Checksums match.
type Checksum uint64

// Params is an abstract set of model parameters.
// It is not thread-safe.
type Params interface {
	// Len returns the length of mutation vectors for
	// this set of parameters.
	Len() int

	// Data serializes the current parameters.
	Data() ([]byte, error)

	// SetData updates the parameters from a piece of
	// serialized data.
	SetData(d []byte) error

	// Update adjusts the parameters by adding the given
	// mutation vector.
	Update(mutation []float64)

	// Checksum generates a checksum for the current set
	// of parameters.
	Checksum() (Checksum, error)
}

// SafeParams is a versioned, thread-safe set of model
// parameters.
// It is similar to Params, except with extra version
// numbers throughout.
//
// Version numbers must increment each time the parameters
// are mutated via SetData or Update.
// Methods which update the version number should return
// the new version number.
type SafeParams interface {
	Len() int
	Data() ([]byte, ParamVersion, error)
	SetData(d []byte) (ParamVersion, error)
	Update(mutation []float64) ParamVersion
	Checksum() (Checksum, ParamVersion, error)
	Version() ParamVersion
}

// MakeSafe synchronizes accesses to p, yielding a safe
// set of parameters.
func MakeSafe(p Params) SafeParams {
	return &safeParams{params: p}
}

type safeParams struct {
	lock    sync.RWMutex
	version ParamVersion
	params  Params
}

func (s *safeParams) Len() int {
	return s.params.Len()
}

func (s *safeParams) Data() (data []byte, version ParamVersion, err error) {
	s.lock.RLock()
	defer s.lock.RUnlock()
	version = s.version
	data, err = s.params.Data()
	return
}

func (s *safeParams) SetData(d []byte) (ParamVersion, error) {
	s.lock.Lock()
	defer s.lock.Unlock()
	err := s.params.SetData(d)
	if err == nil {
		s.version++
	}
	return s.version, err
}

func (s *safeParams) Update(m []float64) ParamVersion {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.params.Update(m)
	s.version++
	return s.version
}

func (s *safeParams) Checksum() (Checksum, ParamVersion, error) {
	s.lock.RLock()
	defer s.lock.RUnlock()
	check, err := s.params.Checksum()
	return check, s.version, err
}

func (s *safeParams) Version() ParamVersion {
	s.lock.RLock()
	defer s.lock.RUnlock()
	return s.version
}

// AnynetParams is a Params implementation that operates
// on a list of *anydiff.Vars.
//
// If you use a Transformer, then the transformer should
// be stateless or implement anysgd.TransformMarshaler.
// This way, the Transformer's internal state remains
// consistent across Slaves, even if new Slaves connect
// after some training has taken place.
type AnynetParams struct {
	Params []*anydiff.Var

	// Transformer, if non-nil, is applied to the updates
	// before adding the updates to the variables.
	Transformer anysgd.Transformer

	// StepSize, if non-zero, is used to scale the updates
	// right before applying them.
	//
	// Since certain types of Transformers are invariant
	// to the magnitude of their inputs, the StepSize
	// parameter might be necessary for those cases.
	StepSize float64
}

// Len returns the total number of parameters across all
// the variables.
func (a *AnynetParams) Len() int {
	var res int
	for _, x := range a.Params {
		res += x.Vector.Len()
	}
	return res
}

// Data serializes the vectors.
func (a *AnynetParams) Data() (data []byte, err error) {
	defer essentials.AddCtxTo("serialize AnynetParams", &err)

	var args []interface{}
	for _, v := range a.Params {
		args = append(args, &anyvecsave.S{Vector: v.Vector})
	}
	if a.Transformer != nil {
		if t, ok := a.Transformer.(anysgd.TransformMarshaler); ok {
			data, err := t.MarshalBinary()
			if err != nil {
				return nil, err
			}
			args = append(args, data)
		}
	}
	return serializer.SerializeAny(args...)
}

// SetData deserializes data into the vectors.
func (a *AnynetParams) SetData(d []byte) (err error) {
	defer essentials.AddCtxTo("deserialize AnynetParams", &err)

	dests := make([]interface{}, len(a.Params))
	for i := range dests {
		dests[i] = new(*anyvecsave.S)
	}
	if a.Transformer != nil {
		if _, ok := a.Transformer.(anysgd.TransformMarshaler); ok {
			dests = append(dests, new([]byte))
		}
	}

	if err := serializer.DeserializeAny(d, dests...); err != nil {
		return err
	}

	for i, v := range a.Params {
		vec := (*dests[i].(**anyvecsave.S)).Vector
		if vec.Len() != v.Vector.Len() {
			return errors.New("length mismatch")
		} else if vec.Creator() != v.Vector.Creator() {
			return errors.New("creator mismatch")
		}
		v.Vector.Set(vec)
	}

	if a.Transformer != nil {
		if t, ok := a.Transformer.(anysgd.TransformMarshaler); ok {
			data := *dests[len(dests)-1].(*[]byte)
			if err := t.UnmarshalBinary(data); err != nil {
				return err
			}
		}
	}

	return nil
}

// Update adds the mutation vector to the parameters by
// splitting it up into sub-vectors for each variable.
func (a *AnynetParams) Update(m []float64) {
	grad := a.SplitMutation(m)
	if a.Transformer != nil {
		grad = a.Transformer.Transform(grad)
	}
	if a.StepSize != 0 {
		for _, v := range grad {
			v.Scale(v.Creator().MakeNumeric(a.StepSize))
		}
	}
	grad.AddToVars()
}

// Checksum computes a checksum by hashing the data
// produced by SetData().
func (a *AnynetParams) Checksum() (ch Checksum, err error) {
	defer essentials.AddCtxTo("checksum AnynetParams", &err)
	data, err := a.Data()
	if err != nil {
		return
	}
	hash := md5.Sum(data)
	for i := uint(0); i < 8; i++ {
		ch |= Checksum(hash[i]) << (i * 8)
	}
	return
}

// SplitMutation splits the mutation vector up into
// separate vectors for each variable.
// It does not apply the Transformer or use the StepSize.
func (a *AnynetParams) SplitMutation(m []float64) anydiff.Grad {
	grad := anydiff.Grad{}
	for _, v := range a.Params {
		subVec := m[:v.Vector.Len()]
		m = m[v.Vector.Len():]
		cr := v.Vector.Creator()
		grad[v] = cr.MakeVectorData(cr.MakeNumericList(subVec))
	}
	if len(m) > 0 {
		panic("index out of bounds")
	}
	return grad
}
