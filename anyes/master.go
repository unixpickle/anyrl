package anyes

import (
	"errors"
	"math/rand"
	"sync"

	"github.com/unixpickle/essentials"
)

// A Master coordinates Slaves to train a model.
//
// The methods on a Master are not all thread-safe.
// In particular, you cannot run Rollouts and Update at
// the same time, nor can you run multiple calls to those
// methods concurrently.
type Master struct {
	// Noise is used to generate mutations.
	Noise *Noise

	// Params is used to store and update the parameters.
	//
	// You should not manually modify parameters once they
	// are in use by a Master.
	Params SafeParams

	// Normalize, if true, indicates that the rewards for
	// each update should be statistically normalized.
	Normalize bool

	// NoiseStddev is the standard deviation for the
	// mutation noise.
	//
	// It is referred to as sigma in the original paper.
	NoiseStddev float64

	// StepSize is the update step size.
	//
	// It is referred to as alpha in the original paper.
	StepSize float64

	// SlaveError is called if a Slave produces an error
	// during a Run or Update call.
	//
	// If this is nil, it is equivalent to an empty method
	// which echoes the error.
	//
	// If SlaveError returns an error, the error will
	// propagate to the method of Master which ultimately
	// caused the error.
	//
	// By the time SlaveError is called, the Slave will
	// have been removed from the Master.
	// SlaveError may re-add the Slave if the error is
	// recoverable.
	SlaveError func(s Slave, err error) error

	slaveLock  sync.RWMutex
	slaves     []*managedSlave
	slaveAdded chan struct{}

	updateLock sync.RWMutex
}

// AddSlave adds a Slave to the pool of Slaves.
// It blocks until the slave has been initialized, or
// returns an error if initialization fails.
//
// After being added, the Slave will automatically be
// utilized by any operations, even ones which are already
// running.
func (m *Master) AddSlave(s Slave) (err error) {
	defer essentials.AddCtxTo("add slave", &err)

	// Ensure that updates don't miss any Slaves which are
	// added right before the update.
	m.updateLock.RLock()
	defer m.updateLock.RUnlock()

	data, version, err := m.Params.Data()
	if err != nil {
		return
	}

	err = s.Init(data, m.Noise.Seed(), m.Noise.Len())
	if err != nil {
		return
	}

	m.slaveLock.Lock()
	m.slaves = append(m.slaves, &managedSlave{
		Slave:   s,
		Version: version,
	})
	if m.slaveAdded != nil {
		select {
		case m.slaveAdded <- struct{}{}:
		default:
		}
	}
	m.slaveLock.Unlock()

	return
}

// Slaves returns a copy of the current list of Slaves.
//
// This is thread-safe.
func (m *Master) Slaves() []Slave {
	m.slaveLock.RLock()
	defer m.slaveLock.RUnlock()
	res := make([]Slave, len(m.slaves))
	for i, s := range m.slaves {
		res[i] = s.Slave
	}
	return res
}

// Rollouts gathers 2*n rollouts from the Slaves.
//
// This blocks until all rollouts are finished or an error
// occurs and is not handled by m.SlaveError.
// If there are no Slaves to utilize, Rollouts will wait
// for Slaves to become available.
//
// In the case of an error, Rollouts waits for all running
// Slaves to complete and returns the partial list of
// results along with the first error encountered.
//
// The stopping conditions are used for every rollout.
// If the stopping conditions are nil, the zero value is
// used.
func (m *Master) Rollouts(stop *StopConds, n int) (rollouts []*Rollout, err error) {
	defer essentials.AddCtxTo("rollouts", &err)

	if stop == nil {
		stop = &StopConds{}
	}

	jobs := make(chan *scaleSeed, n*2)
	for i := 0; i < n; i++ {
		seed := rand.Int63()
		for _, scale := range []float64{-1, 1} {
			jobs <- &scaleSeed{Scale: scale * m.NoiseStddev, Seed: seed}
		}
	}

	resChan := make(chan *Rollout, n*2)
	errChan := make(chan error, 1)

	handledErrChan := make(chan struct{}, 1)
	slaveAddedChan := m.getSlaveAdded()

	var wg sync.WaitGroup
	for len(rollouts) < n*2 {
		assignments := m.assignJobs(jobs)

		for _, assig := range assignments {
			wg.Add(1)
			go func(assig *jobAssignment) {
				defer wg.Done()
				r, err := assig.Slave.Slave.Run(stop, assig.Job.Scale, assig.Job.Seed)
				if err != nil {
					jobs <- assig.Job
					m.removeSlave(assig.Slave)
					err = m.callSlaveError(assig.Slave.Slave, err)
					if err != nil {
						select {
						case errChan <- err:
						default:
						}
					} else {
						select {
						case handledErrChan <- struct{}{}:
						default:
						}
					}
				} else {
					assig.Slave.WorkingLock.Lock()
					assig.Slave.Working = false
					assig.Slave.WorkingLock.Unlock()
					resChan <- r
				}
			}(assig)
		}

		select {
		case err := <-errChan:
			go func() {
				wg.Wait()
				close(resChan)
			}()
			for rollout := range resChan {
				rollouts = append(rollouts, rollout)
			}
			return rollouts, err
		case <-handledErrChan:
			// Attempt to assign the new job.
		case <-slaveAddedChan:
			// Attempt to use the new slave.
		case r := <-resChan:
			rollouts = append(rollouts, r)
		}
	}

	// Deal with race condition:
	//
	// If one job sends to resChan while another sends to
	// errChan, select might read from resChan, then the
	// next iteration of the loop might re-schedule the
	// job and get the final result.
	select {
	case err = <-errChan:
	default:
	}

	return
}

// Update updates the parameters using the rollouts and
// pushes the updates to all the other slaves.
//
// Update returns the first error encountered, but runs
// until all the Slaves have attempted the update (even
// if some of them failed).
func (m *Master) Update(r []*Rollout) (err error) {
	defer essentials.AddCtxTo("update", &err)

	scales, seeds := m.scalesAndSeeds(r)

	oldVerison := m.Params.Version()
	newVersion := m.localUpdate(scales, seeds)

	errChan := make(chan error, 1)

	var wg sync.WaitGroup
	m.slaveLock.RLock()
	for _, slave := range m.slaves {
		if slave.Version == newVersion {
			// Can happen if m.AddSlave added the slave
			// right after localUpdate finished.
			continue
		} else if slave.Version != oldVerison {
			m.slaveLock.RUnlock()
			return errors.New("parameter version inconsistency")
		}
		wg.Add(1)
		go func(slave *managedSlave) {
			defer wg.Done()
			if err := slave.Slave.Update(scales, seeds); err != nil {
				m.removeSlave(slave)
				err = m.callSlaveError(slave.Slave, err)
				if err != nil {
					select {
					case errChan <- err:
					default:
					}
				}
			} else {
				slave.Version = newVersion
			}
		}(slave)
	}
	m.slaveLock.RUnlock()

	wg.Wait()

	close(errChan)
	return <-errChan
}

func (m *Master) localUpdate(scales []float64, seeds []int64) ParamVersion {
	m.updateLock.Lock()
	defer m.updateLock.Unlock()

	vec := m.Noise.GenSum(scales, seeds, m.Params.Len())
	return m.Params.Update(vec)
}

// assignJobs assigns pending jobs to idle slaves.
// It automatically changes the slaves' working status.
func (m *Master) assignJobs(jobs chan *scaleSeed) []*jobAssignment {
	m.slaveLock.RLock()
	defer m.slaveLock.RUnlock()

	var res []*jobAssignment

	perm := rand.Perm(len(m.slaves))
	for _, i := range perm {
		slave := m.slaves[i]
		slave.WorkingLock.Lock()
		if slave.Working {
			slave.WorkingLock.Unlock()
			continue
		}
		select {
		case job := <-jobs:
			slave.Working = true
			slave.WorkingLock.Unlock()
			res = append(res, &jobAssignment{Slave: slave, Job: job})
		default:
			slave.WorkingLock.Unlock()
			return res
		}
	}

	return res
}

func (m *Master) scalesAndSeeds(r []*Rollout) ([]float64, []int64) {
	var scales []float64
	var seeds []int64
	for _, rollout := range r {
		scales = append(scales, rollout.Reward)
		seeds = append(seeds, rollout.Seed)
	}
	if m.Normalize {
		normalize(scales)
	}

	// We square m.NoiseStddev to cancel out the sigma
	// from rollout.Scale as well as capture a 1/sigma
	// from the original paper's formulation.
	globalScale := m.StepSize / (m.NoiseStddev * m.NoiseStddev * float64(len(r)))

	for i, rollout := range r {
		scales[i] *= globalScale * rollout.Scale
	}
	return scales, seeds
}

func (m *Master) removeSlave(toRemove *managedSlave) {
	m.slaveLock.Lock()
	defer m.slaveLock.Unlock()
	for i, s := range m.slaves {
		if s == toRemove {
			m.slaves[i] = m.slaves[len(m.slaves)-1]

			// Avoid maintaining a reference to the deleted
			// slave instance.
			m.slaves[len(m.slaves)-1] = nil

			m.slaves = m.slaves[:len(m.slaves)-1]
		}
	}
}

func (m *Master) callSlaveError(s Slave, err error) error {
	if m.SlaveError != nil {
		return m.SlaveError(s, err)
	} else {
		return err
	}
}

func (m *Master) getSlaveAdded() chan struct{} {
	m.slaveLock.Lock()
	defer m.slaveLock.Unlock()
	if m.slaveAdded == nil {
		m.slaveAdded = make(chan struct{}, 1)
	}
	return m.slaveAdded
}

type managedSlave struct {
	Slave   Slave
	Version ParamVersion

	// Set by Rollouts.
	WorkingLock sync.Mutex
	Working     bool
}

type scaleSeed struct {
	Scale float64
	Seed  int64
}

type jobAssignment struct {
	Slave *managedSlave
	Job   *scaleSeed
}
