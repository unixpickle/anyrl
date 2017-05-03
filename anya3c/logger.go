package anya3c

import (
	"log"

	"github.com/unixpickle/anyvec"
)

// A Logger logs status messages which are produced during
// A3C training.
type Logger interface {
	LogEpisode(workerID int, reward float64)
	LogUpdate(workerID int)
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
func (s *StandardLogger) LogUpdate(workerID int) {
	if s.Update {
		log.Printf("update: worker=%d", workerID)
	}
}

// LogRegularization logs the regularization term for an
// action distribution.
func (s *StandardLogger) LogRegularize(workerID int, term anyvec.Numeric) {
	if s.Regularize {
		log.Printf("regularize: worker=%d term=%v", workerID, term)
	}
}
