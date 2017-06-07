package anyes

import "math/rand"

// Noise produces deterministic Gaussian noise.
type Noise struct {
	seed   int64
	data   []float64
	lenLog int
}

// NewNoise creates a noise generator with the given seed
// and amount of pre-generated noise.
//
// The length argument must be a power of 2.
func NewNoise(seed int64, length int) *Noise {
	gen := rand.New(rand.NewSource(seed))
	var data []float64
	for i := 0; i < length; i++ {
		data = append(data, gen.NormFloat64())
	}
	return &Noise{seed: seed, data: data, lenLog: log2(length)}
}

// Gen generates a chunk of noise in a deterministic way
// based on the given seed.
// The noise is scaled by the given factor.
func (n *Noise) Gen(scale float64, seed int64, amount int) []float64 {
	source := rand.NewSource(seed)
	res := make([]float64, amount)
	for i, sourceIdx := range n.randomIndices(source, amount) {
		res[i] = scale * n.data[sourceIdx]
	}
	return res
}

// GenSum generates multiple chunks of noise (given by the
// seeds), scales each chunk by the corresponding scale,
// and sums the result.
func (n *Noise) GenSum(scales []float64, seeds []int64, amount int) []float64 {
	if len(scales) != len(seeds) {
		panic("mismatching number of scales and seeds")
	}
	res := make([]float64, amount)
	for i, scale := range scales {
		seed := seeds[i]
		for i, x := range n.Gen(scale, seed, amount) {
			res[i] += x
		}
	}
	return res
}

// Seed returns the seed needed to create an identical
// Noise with NewNoise().
func (n *Noise) Seed() int64 {
	return n.seed
}

// Len returns the amount of pre-generated noise.
// This is necessary when creating a new noise generator
// with NewNoise().
func (n *Noise) Len() int {
	return len(n.data)
}

func (n *Noise) randomIndices(source rand.Source, size int) []int {
	res := make([]int, 0, size+64)
	mask := int64((1 << uint(n.lenLog)) - 1)

	for len(res) < size {
		next := source.Int63()
		for remaining := 63; remaining >= n.lenLog; remaining -= n.lenLog {
			res = append(res, int(next&mask))
			next >>= uint(n.lenLog)
		}
	}

	return res[:size]
}

func log2(n int) int {
	for i := uint(0); i < 64; i++ {
		exp := uint64(1) << i
		if exp == uint64(n) {
			return int(i)
		} else if exp > uint64(n) {
			break
		}
	}
	panic("not a power of 2")
}
