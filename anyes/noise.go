package anyes

import (
	"math/rand"
	"reflect"
	"sync"
)

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

// A NoiseGroup wraps a Noise instance and caches results
// so that multiple slaves on one machine can utilize the
// same generated noise.
type NoiseGroup struct {
	lock     sync.Mutex
	noise    *Noise
	scales   []float64
	seeds    []int64
	amount   int
	doneChan chan []float64
}

// Init initializes the group.
// If it is called multiple times, all later calls must
// have the same arguments as the first one.
func (n *NoiseGroup) Init(seed int64, length int) {
	n.lock.Lock()
	defer n.lock.Unlock()
	if n.noise == nil {
		n.noise = NewNoise(seed, length)
	} else {
		if n.noise.Seed() != seed || n.noise.Len() != length {
			panic("different noise arguments on same NoiseGroup")
		}
	}
}

// Gen generates noise without caching.
func (n *NoiseGroup) Gen(scale float64, seed int64, amount int) []float64 {
	return n.noise.Gen(scale, seed, amount)
}

// GenSum generates a linear combination of noise vectors
// with caching.
//
// The returned noise is a copy of the cache, so the
// caller may modify it at will.
func (n *NoiseGroup) GenSum(scales []float64, seeds []int64, amount int) []float64 {
	n.lock.Lock()
	if n.amount == amount && reflect.DeepEqual(n.scales, scales) &&
		reflect.DeepEqual(n.seeds, seeds) {
		ch := n.doneChan
		n.lock.Unlock()
		res := <-ch
		ch <- res
		return append([]float64{}, res...)
	} else {
		ch := make(chan []float64, 1)
		n.scales = scales
		n.seeds = seeds
		n.amount = amount
		n.doneChan = ch
		n.lock.Unlock()
		res := n.noise.GenSum(scales, seeds, amount)
		ch <- append([]float64{}, res...)
		return res
	}
}
