package anya3c

import (
	"log"
	"sync"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

// A Logger logs status messages which are produced during
// A3C training.
type Logger interface {
	LogEpisode(workerID int, reward float64)
	LogUpdate(workerID int, criticMSE anyvec.Numeric)
	LogRegularize(workerID int, term anyvec.Numeric)
}

// StandardLogger is a Logger which uses the log package.
//
// A Field of name <N> controls whether or not the Log<N>
// method does anything.
type StandardLogger struct {
	Episode    bool
	Update     bool
	Regularize bool
}

// LogEpisode logs the result of an episode.
func (s *StandardLogger) LogEpisode(workerID int, reward float64) {
	if s.Episode {
		log.Printf("episode: worker=%d reward=%f", workerID, reward)
	}
}

// LogUpdate logs the fact that a step was taken.
func (s *StandardLogger) LogUpdate(workerID int, criticMSE anyvec.Numeric) {
	if s.Update {
		log.Printf("update: worker=%d critic_mse=%f", workerID, criticMSE)
	}
}

// LogRegularization logs the regularization term for an
// action distribution.
func (s *StandardLogger) LogRegularize(workerID int, term anyvec.Numeric) {
	if s.Regularize {
		log.Printf("regularize: worker=%d term=%v", workerID, term)
	}
}

// AvgLogger logs averages over multiple finer-grained log
// messages.
// It is useful for reducing the number of log messages in
// high-speed training environments.
//
// Every integer field <N> is the average size for the log
// routine called Log<N>.
// When an average size is 0, its corresponding logs are
// forwarded directly without averaging.
type AvgLogger struct {
	// Logger is where averaged logs are forwarded.
	Logger Logger

	// Creator is used to apply numerical operations
	// to anyvec.Numerics.
	Creator anyvec.Creator

	Episode    int
	Update     int
	Regularize int

	episodeAvg    averager
	updateAvg     averager
	regularizeAvg averager
}

// LogEpisode logs the result of an episode.
func (a *AvgLogger) LogEpisode(workerID int, reward float64) {
	cr := anyvec64.DefaultCreator{}
	if avg := a.episodeAvg.Add(a.Episode, cr, reward); avg != nil {
		a.Logger.LogEpisode(workerID, avg.(float64))
	}
}

// LogUpdate logs the fact that a step was taken.
func (a *AvgLogger) LogUpdate(workerID int, criticMSE anyvec.Numeric) {
	if avg := a.updateAvg.Add(a.Update, a.Creator, criticMSE); avg != nil {
		a.Logger.LogUpdate(workerID, avg)
	}
}

// LogRegularization logs the regularization term for an
// action distribution.
func (a *AvgLogger) LogRegularize(workerID int, term anyvec.Numeric) {
	if avg := a.regularizeAvg.Add(a.Regularize, a.Creator, term); avg != nil {
		a.Logger.LogRegularize(workerID, avg)
	}
}

type averager struct {
	Lock     sync.Mutex
	CurCount int
	CurSum   anyvec.Numeric
}

// Add adds to the average and returns non-nil when the
// average is complete.
func (a *averager) Add(avgSize int, creator anyvec.Creator,
	num anyvec.Numeric) anyvec.Numeric {
	if avgSize == 0 {
		return num
	}
	a.Lock.Lock()
	defer a.Lock.Unlock()
	a.CurCount++
	ops := creator.NumOps()
	if a.CurSum == nil {
		a.CurSum = num
	} else {
		a.CurSum = ops.Add(a.CurSum, num)
	}
	if a.CurCount == avgSize {
		avg := ops.Div(a.CurSum, creator.MakeNumeric(float64(avgSize)))
		a.CurCount = 0
		a.CurSum = creator.MakeNumeric(0)
		return avg
	}
	return nil
}
