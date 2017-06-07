package anyes

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/unixpickle/approb"
)

func TestNoiseRandomness(t *testing.T) {
	noise := NewNoise(0, 1<<3)
	for i := range noise.data {
		noise.data[i] = float64(i)
	}

	corr := approb.Correlation(20000, 0.1, func() float64 {
		return 0.3 * float64(rand.Intn(1<<3))
	}, func() float64 {
		noise := noise.Gen(0.3, rand.Int63(), 5)
		return noise[rand.Intn(len(noise))]
	})

	if math.Abs(corr-1) > 1e-3 {
		t.Errorf("correlation should be 1 but got %f", corr)
	}
}

func TestNoiseDeterminism(t *testing.T) {
	noise := NewNoise(1234, 1<<15)
	data := noise.Gen(0.5, 1222, 30)
	noise = NewNoise(1234, 1<<15)

	// Make sure it's not stateful.
	noise.Gen(0.3, 13333, 10)

	data1 := noise.Gen(0.5, 1222, 30)

	if !reflect.DeepEqual(data1, data) {
		t.Error("non-deterministic noise")
	}
}

func BenchmarkNoiseGen(b *testing.B) {
	noise := NewNoise(1337, 1<<15)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		noise.Gen(0.5, 1234, 1<<22)
	}
}
