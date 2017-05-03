package anyrl

import (
	"math"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
)

// TotalRewards sums the rewards for each sequence of
// rewards.
func TotalRewards(c anyvec.Creator, rewards lazyseq.Tape) anyvec.Vector {
	sum := TotalRewardsBatch(rewards)
	if sum == nil {
		return c.MakeVector(0)
	}
	return sum.Packed
}

// TotalRewardsBatch is like TotalRewards except that it
// preserves the sequence present map.
//
// This returns nil if there are no timesteps.
func TotalRewardsBatch(rewards lazyseq.Tape) *anyseq.Batch {
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

// MeanReward sums the rewards for each sequence, then
// computes the mean of the sums.
func MeanReward(c anyvec.Creator, rewards lazyseq.Tape) anyvec.Numeric {
	total := TotalRewards(c, rewards)
	return sampleMean(total)
}

// RewardVariance computes the variance of total rewards.
func RewardVariance(c anyvec.Creator, rewards lazyseq.Tape) anyvec.Numeric {
	total := TotalRewards(c, rewards)
	negMean := c.NumOps().Mul(sampleMean(total), c.MakeNumeric(-1))
	total.AddScalar(negMean)

	// Second moment of centered samples is the variance.
	anyvec.Pow(total, c.MakeNumeric(2))
	return sampleMean(total)
}

// DiscountedRewards computes discounted rewards.
func DiscountedRewards(rewards lazyseq.Tape, factor float64) lazyseq.Tape {
	res, writer := lazyseq.ReferenceTape()

	var i float64
	for in := range rewards.ReadTape(0, -1) {
		discount := math.Pow(factor, i)
		scaled := in.Reduce(in.Present)
		scaled.Packed.Scale(scaled.Packed.Creator().MakeNumeric(discount))
		writer <- scaled
		i++
	}

	close(writer)
	return res
}

func sampleMean(vec anyvec.Vector) anyvec.Numeric {
	c := vec.Creator()
	return c.NumOps().Div(anyvec.Sum(vec), c.MakeNumeric(float64(vec.Len())))
}
