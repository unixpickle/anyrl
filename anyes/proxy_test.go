package anyes

import (
	"errors"
	"fmt"
	"io"
	"reflect"
	"testing"
	"time"

	"github.com/unixpickle/essentials"
)

func TestProxy(t *testing.T) {
	slave := &testSlave{}
	pipe1, pipe2 := bidirPipe()
	defer pipe1.Close()
	defer pipe2.Close()

	go ProxyProvide(pipe1, slave)

	proxy, err := ProxyConsume(pipe2)
	if err != nil {
		t.Fatal(err)
	}
	defer proxy.Close()

	if err != nil {
		t.Fatal(err)
	}

	err = proxy.Init([]byte("hi"), 15, 1337)
	if err != nil {
		t.Fatal(err)
	}
	err = verifyArgs(slave, []byte("hi"), int64(15), 1337)
	if err != nil {
		t.Error(err)
	}

	slave.retErr = errors.New("hello world!")
	err = verifyError(slave, proxy.Init([]byte("hey"), 15, 1337))
	if err != nil {
		t.Error(err)
	}

	slave.retErr = nil
	slave.retRollout = &Rollout{
		Scale:     3.5,
		Seed:      666,
		Reward:    9001,
		Steps:     120,
		EarlyStop: true,
	}
	rollout, err := proxy.Run(&StopConds{MaxTime: time.Minute}, 3.5, 666)
	if err != nil {
		t.Fatal(err)
	}
	err = verifyArgs(slave, &StopConds{MaxTime: time.Minute}, 3.5, int64(666))
	if err != nil {
		t.Error(err)
	}
	if !reflect.DeepEqual(slave.retRollout, rollout) {
		t.Errorf("expected rollout %v but got %v", slave.retRollout, rollout)
	}

	slave.retErr = errors.New("run, world!")
	_, err = proxy.Run(&StopConds{}, 3.5, 666)
	err = verifyError(slave, err)
	if err != nil {
		t.Error(err)
	}

	slave.retErr = nil
	err = proxy.Update([]float64{1, 2, 3}, []int64{4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	err = verifyArgs(slave, []float64{1, 2, 3}, []int64{4, 5, 6})
	if err != nil {
		t.Error(err)
	}

	slave.retErr = errors.New("update, world!")
	err = verifyError(slave, proxy.Update([]float64{1, 2, 3}, []int64{4, 5, 6}))
	if err != nil {
		t.Error(err)
	}
}

func verifyError(slave *testSlave, err error) error {
	if err.(*essentials.CtxError).Original.Error() != slave.retErr.Error() {
		return fmt.Errorf("expected error %#v but got %#v",
			slave.retErr.Error(),
			err.(*essentials.CtxError).Original.Error())
	}
	return nil
}

func verifyArgs(slave *testSlave, args ...interface{}) error {
	if !reflect.DeepEqual(args, slave.lastArgs) {
		return fmt.Errorf("expected arguments %v but got %v", args, slave.lastArgs)
	}
	return nil
}

type testSlave struct {
	lastArgs   []interface{}
	retErr     error
	retRollout *Rollout
}

func (t *testSlave) Init(a1 []byte, a2 int64, a3 int) error {
	t.lastArgs = []interface{}{a1, a2, a3}
	return t.retErr
}

func (t *testSlave) Run(a1 *StopConds, a2 float64, a3 int64) (*Rollout, error) {
	t.lastArgs = []interface{}{a1, a2, a3}
	return t.retRollout, t.retErr
}

func (t *testSlave) Update(a1 []float64, a2 []int64) error {
	t.lastArgs = []interface{}{a1, a2}
	return t.retErr
}

func bidirPipe() (*readerWriter, *readerWriter) {
	r1, w1 := io.Pipe()
	r2, w2 := io.Pipe()
	return &readerWriter{r1, w2}, &readerWriter{r2, w1}
}

type readerWriter struct {
	reader io.ReadCloser
	writer io.WriteCloser
}

func (b *readerWriter) Read(d []byte) (int, error) {
	return b.reader.Read(d)
}

func (b *readerWriter) Write(d []byte) (int, error) {
	return b.writer.Write(d)
}

func (b *readerWriter) Close() error {
	b.reader.Close()
	b.writer.Close()
	return nil
}
