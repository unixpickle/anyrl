package anya3c

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestBPTT(t *testing.T) {
	c := anyvec64.DefaultCreator{}

	agent := &LocalAgent{
		Agent: &Agent{
			Base: anyrnn.NewLSTM(c, 3, 4),
			Actor: anyrnn.Stack{
				anyrnn.NewLSTM(c, 4, 2),
				&anyrnn.LayerBlock{
					Layer: anynet.NewAffine(c, 5, 0),
				},
			},
			Critic: anyrnn.NewLSTM(c, 4, 1),

			ActionSpace: anyrl.Softmax{},
		},
	}
	agent.Params = agent.AllParameters()

	sampledVecs := []anyvec.Vector{
		c.MakeVectorData([]float64{1, 0}),
		c.MakeVectorData([]float64{1, 0}),
		c.MakeVectorData([]float64{0, 1}),
	}
	sampledSeq := anyseq.ConstSeqList(c, [][]anyvec.Vector{sampledVecs})

	inputs := anyseq.ConstSeqList(c, [][]anyvec.Vector{
		{
			c.MakeVectorData([]float64{1, -1, 0}),
			c.MakeVectorData([]float64{-0.2, 0.5, -0.5}),
			c.MakeVectorData([]float64{-1, 0, 1}),
		},
	})
	rewards := anyseq.ConstSeqList(c, [][]anyvec.Vector{
		{
			c.MakeVectorData([]float64{1 + 0.4*(0.5+0.4*-0.7)}),
			c.MakeVectorData([]float64{0.5 + 0.4*-0.7}),
			c.MakeVectorData([]float64{-0.7}),
		},
	})

	actorBlock := anyrnn.Stack{
		agent.Base,
		agent.Actor,
		&anyrnn.LayerBlock{Layer: anynet.LogSoftmax},
	}
	actorOut := anyrnn.Map(inputs, actorBlock)

	criticBlock := anyrnn.Stack{agent.Base, agent.Critic}
	criticOut := anyrnn.Map(inputs, criticBlock)

	criticTerm := anyseq.Sum(anyseq.MapN(func(n int, v ...anydiff.Res) anydiff.Res {
		target := v[0]
		output := v[1]
		return anydiff.Scale(
			anydiff.Pow(anydiff.Sub(target, output), c.MakeNumeric(2)),
			c.MakeNumeric(-1),
		)
	}, rewards, criticOut))
	actorTerm := anyseq.Sum(anyseq.MapN(func(n int, v ...anydiff.Res) anydiff.Res {
		target := v[0]
		criticOut := v[1]
		advantage := anydiff.NewConst(anydiff.Sub(target, criticOut).Output())
		actorOut := v[2]
		sampled := v[3]
		entropy := anydiff.Dot(anydiff.Exp(actorOut), actorOut)
		return anydiff.Add(
			anydiff.Mul(anydiff.Dot(actorOut, sampled), advantage),
			anydiff.Scale(entropy, c.MakeNumeric(-0.5)),
		)
	}, rewards, criticOut, actorOut, sampledSeq))

	expectedGrad := anydiff.NewGrad(agent.Params...)
	totalError := anydiff.Add(criticTerm, actorTerm)
	totalError.Propagate(c.MakeVectorData([]float64{1}), expectedGrad)

	rollout := &rollout{
		Beginning: true,
		Rewards:   []float64{1, 0.5, -0.7},
		Sampled:   sampledVecs,
	}
	worker := &worker{
		Agent: agent,
		AgentState: []anyrnn.State{
			agent.Base.Start(1),
			agent.Actor.Start(1),
			agent.Critic.Start(1),
		},
	}
	for _, input := range inputs.Output() {
		worker.EnvObs = input.Packed
		worker.StepAgent()
		rollout.Outs = append(rollout.Outs, worker.AgentRes)
		worker.AgentRes = nil
	}
	worker.EnvDone = true

	b := &bptt{
		Rollout:  rollout,
		Worker:   worker,
		Discount: 0.4,
		Regularizer: &anypg.EntropyReg{
			Entropyer: anyrl.Softmax{},
			Coeff:     0.5,
		},
	}
	actualGrad := b.Run()

	if len(actualGrad) != len(expectedGrad) {
		t.Fatalf("expected %d gradients but got %d", len(expectedGrad),
			len(actualGrad))
	}

	for key, x := range expectedGrad {
		a := actualGrad[key]
		diff := a.Copy()
		diff.Sub(x)
		if anyvec.AbsMax(diff).(float64) > 1e-3 {
			t.Errorf("bad vector: %v (expected %v)", a.Data(), x.Data())
		}
	}
}
