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
	sum := rewardSum(rewards)

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
	sum := rewardSum(rewards)
	if sum == nil {
		return c.MakeVector(0)
	}
	return sum.Packed
}

// MeanReward sums the rewards for each sequence, then
// computes the mean of the sums.
func MeanReward(c anyvec.Creator, rewards lazyseq.Tape) anyvec.Numeric {
	total := TotalRewards(c, rewards)
	total.Scale(total.Creator().MakeNumeric(1 / float64(total.Len())))
	return anyvec.Sum(total)
}

// RepeatedTotalRewards transforms a reward sequence into
// a sequence where the total reward reappears at every
// timestep.
//
// If normalize is true, then the total rewards are
// statistically normalized to have mean 0 and standard
// deviation 1.
func RepeatedTotalRewards(rewards lazyseq.Tape, normalize bool) lazyseq.Tape {
	sum := rewardSum(rewards)
	if sum == nil {
		return rewards
	}

	if normalize {
		normalizeVec(sum.Packed)
	}

	resTape, writer := lazyseq.ReferenceTape()
	for batch := range rewards.ReadTape(0, -1) {
		writer <- sum.Reduce(batch.Present)
	}
	close(writer)
	return resTape

}

func rewardSum(rewards lazyseq.Tape) *anyseq.Batch {
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
	return sum
}

func normalizeVec(vec anyvec.Vector) {
	c := vec.Creator()

	// Set mean=0 so we can use second moment as variance.
	meanVals := vec.Copy()
	meanVals.Scale(c.MakeNumeric(-1 / float64(vec.Len())))
	vec.AddScalar(anyvec.Sum(meanVals))

	stdVals := vec.Copy()
	anyvec.Pow(stdVals, c.MakeNumeric(2))
	stdVals.Scale(c.MakeNumeric(1 / float64(vec.Len())))
	secondMoment := anyvec.Sum(stdVals)
	invStdVec := c.MakeVector(1)
	invStdVec.AddScalar(secondMoment)
	anyvec.Pow(invStdVec, c.MakeNumeric(-0.5))
	invStd := anyvec.Sum(invStdVec)

	vec.Scale(invStd)
}
