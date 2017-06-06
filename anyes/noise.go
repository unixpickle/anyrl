package anyes

import "math/rand"

// Noise produces deterministic Gaussian noise.
type Noise struct {
	seed int64
	data []float64
}

// NewNoise creates a noise generator with the given seed
// and amount of pre-generated noise.
func NewNoise(seed int64, length int) *Noise {
	gen := rand.New(rand.NewSource(seed))
	var data []float64
	for i := 0; i < length; i++ {
		data = append(data, gen.NormFloat64())
	}
	return &Noise{seed: seed, data: data}
}

// Gen generates a chunk of noise in a deterministic way
// based on the given seed.
func (n *Noise) Gen(seed int64, amount int) []float64 {
	gen := rand.New(rand.NewSource(seed))
	res := make([]float64, amount)
	for i := 0; i < amount; i++ {
		res[i] = n.data[gen.Intn(len(n.data))]
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
		for i, x := range n.Gen(seed, amount) {
			res[i] += scale * x
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
