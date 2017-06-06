package anyes

import "sync"

// ParamVersion is a version number used by SafeParams.
type ParamVersion int64

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

func (s *safeParams) Version() ParamVersion {
	s.lock.RLock()
	defer s.lock.RUnlock()
	return s.version
}
