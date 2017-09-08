package anyrl

import (
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
)

// Rewards is a set of reward sequences.
// Each sequence can be thought of as a different episode.
type Rewards [][]float64

// PackRewards concatenates Rewards objects.
func PackRewards(r []Rewards) Rewards {
	var res Rewards
	for _, x := range r {
		res = append(res, x...)
	}
	return res
}

// Tape converts the reward sequences to a lazyseq.Tape.
//
// If all the reward sequences are empty, then c is never
// used and may be nil.
func (r Rewards) Tape(c anyvec.Creator) lazyseq.Tape {
	res, writer := lazyseq.ReferenceTape(c)

	var t int
	for {
		present := make([]bool, len(r))
		var packed []float64
		for seqIdx, seq := range r {
			if t < len(seq) {
				present[seqIdx] = true
				packed = append(packed, seq[t])
			}
		}
		if len(packed) == 0 {
			break
		}
		writer <- &anyseq.Batch{
			Packed:  c.MakeVectorData(c.MakeNumericList(packed)),
			Present: present,
		}
		t++
	}

	close(writer)
	return res
}

// Totals returns the reward sum for each sequence.
func (r Rewards) Totals() []float64 {
	sums := make([]float64, len(r))
	for seqIdx, seq := range r {
		for _, x := range seq {
			sums[seqIdx] += x
		}
	}
	return sums
}

// Mean computes the mean of the reward totals.
func (r Rewards) Mean() float64 {
	var sum float64
	for _, x := range r.Totals() {
		sum += x
	}
	return sum / float64(len(r))
}

// Variance computes the variance of the reward totals.
func (r Rewards) Variance() float64 {
	var sum float64
	var sqSum float64
	for _, x := range r.Totals() {
		sum += x
		sqSum += x * x
	}

	mean := sum / float64(len(r))
	secondMoment := sqSum / float64(len(r))

	return secondMoment - mean*mean
}

// Reduce produces a new Rewards object where certain
// sequences have been removed (set to nil).
//
// If pres[i] is false, then the i-th sequence is removed.
func (r Rewards) Reduce(pres []bool) Rewards {
	res := make(Rewards, len(r))
	for i, p := range pres {
		if p {
			res[i] = r[i]
		}
	}
	return res
}
