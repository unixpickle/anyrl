package anyrl

import (
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
)

// RemainingRewards transforms a sequence of rewards so
// that each reward is replaced with the sum of all the
// rewards from that timestep to the end of the episode.
//
// This can be used to approximate the state-action value
// function.
func RemainingRewards(rewards lazyseq.Tape) lazyseq.Tape {
	var sum *anyseq.Batch
	for batch := range rewards.ReadTape(0, -1) {
		if sum == nil {
			sum = &anyseq.Batch{
				Present: batch.Present,
				Packed:  batch.Packed.Copy(),
			}
		} else {
			sum.Packed.Add(batch.Expand(sum.Present).Packed)
		}
	}

	resTape, writer := lazyseq.ReferenceTape()

	for batch := range rewards.ReadTape(0, -1) {
		// Create two separate copies of the sum to
		// avoid modifying the batch we send.
		writer <- sum.Reduce(batch.Present)
		sum = sum.Reduce(batch.Present)
		sum.Packed.Sub(batch.Packed)
	}

	close(writer)
	return resTape
}

// TotalRewards sums the rewards for each sequence of
// rewards.
func TotalRewards(c anyvec.Creator, rewards lazyseq.Tape) anyvec.Vector {
	return lazyseq.SumEach(lazyseq.TapeRereader(c, rewards)).Output()
}

// MeanReward sums the rewards for each sequence, then
// computes the mean of the sums.
func MeanReward(c anyvec.Creator, rewards lazyseq.Tape) anyvec.Numeric {
	total := TotalRewards(c, rewards)
	total.Scale(total.Creator().MakeNumeric(1 / float64(total.Len())))
	return anyvec.Sum(total)
}

// CenterRewards subtracts the mean reward.
func CenterRewards(c anyvec.Creator, rewards lazyseq.Tape) lazyseq.Tape {
	mean := MeanReward(c, rewards)
	negMeanVec := c.MakeVector(1)
	negMeanVec.AddScalar(mean)
	negMeanVec.Scale(c.MakeNumeric(-1))
	negMean := anyvec.Sum(negMeanVec)

	resTape, writer := lazyseq.ReferenceTape()

	for batch := range rewards.ReadTape(0, -1) {
		newBatch := &anyseq.Batch{
			Present: batch.Present,
			Packed:  batch.Packed.Copy(),
		}
		newBatch.Packed.AddScalar(negMean)
		writer <- newBatch
	}

	close(writer)
	return resTape
}
