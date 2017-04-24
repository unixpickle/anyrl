package anyrl

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyfwd"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/serializer"
)

// Default number of iterations for Conjugate Gradients.
const DefaultConjGradIters = 10

// NaturalPG implements natural policy gradients.
// Due to requirements involivng second derivatives,
// NaturalPG requires more detailed access to the policy
// than does PolicyGradient.
type NaturalPG struct {
	Policy      anyrnn.Block
	Params      []*anydiff.Var
	ActionSpace ActionSpace

	// Iters specifies the number of iterations of the
	// Conjugate Gradients algorithm.
	// If 0, DefaultConjGradIters is used.
	Iters int

	// ApplyPolicy applies a policy to an input sequence.
	// If nil, back-propagation through time is used.
	ApplyPolicy func(s lazyseq.Rereader, b anyrnn.Block) lazyseq.Rereader

	// ActionJudger is used to judge actions.
	//
	// If nil, TotalJudger is used.
	ActionJudger ActionJudger

	// Reduce is used to decide which rollouts to use when
	// solving for the natural gradient.
	//
	// For an example of something to use, see FracReducer.
	//
	// If nil, all rollouts are used.
	Reduce func(in *RolloutSet) *RolloutSet
}

// Run computes the natural gradient for the rollouts.
func (n *NaturalPG) Run(r *RolloutSet) anydiff.Grad {
	var seq lazyseq.Reuser
	pg := &PG{
		Policy: func(in lazyseq.Rereader) lazyseq.Rereader {
			seq = lazyseq.MakeReuser(n.apply(in, n.Policy))
			return seq
		},
		Params:       n.Params,
		LogProber:    n.ActionSpace,
		ActionJudger: n.ActionJudger,
	}
	grad := pg.Run(r)

	// We check for an all-zero gradient because that is
	// a fairly common case (if all rollouts were optimal,
	// the gradient may be 0).
	if len(grad) == 0 || allZeros(grad) {
		return grad
	}

	if n.Reduce != nil {
		c := creatorFromGrad(grad)
		r = n.Reduce(r)
		in := lazyseq.TapeRereader(c, r.Inputs)
		seq = lazyseq.MakeReuser(n.apply(in, n.Policy))
	}

	n.conjugateGradients(r, seq, grad)

	return grad
}

func (n *NaturalPG) conjugateGradients(r *RolloutSet, policyOuts lazyseq.Reuser,
	grad anydiff.Grad) {
	c := creatorFromGrad(grad)
	ops := c.NumOps()

	// Solving "Fx = grad" for x, where F is the
	// Fisher matrix.
	// Algorithm taken from
	// https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm.

	// x = 0
	x := zeroGrad(grad)

	// r = b - Ax = b
	residual := copyGrad(grad)

	// p = r
	proj := copyGrad(grad)

	residualMag := dotGrad(residual, residual)

	for i := 0; i < n.iters(); i++ {
		// A*p
		policyOuts.Reuse()
		appliedProj := n.applyFisher(r, proj, policyOuts)

		// (r dot r) / (p dot A*p)
		alpha := ops.Div(residualMag, dotGrad(proj, appliedProj))

		// x = x + alpha*p
		alphaProj := copyGrad(proj)
		alphaProj.Scale(alpha)
		addToGrad(x, alphaProj)

		// r = r - alpha*A*p
		appliedProj.Scale(alpha)
		subFromGrad(residual, appliedProj)

		// (newR dot newR) / (r dot r)
		newResidualMag := dotGrad(residual, residual)
		beta := ops.Div(newResidualMag, residualMag)
		residualMag = newResidualMag

		// p = beta*p + r
		oldProj := proj
		proj = copyGrad(residual)
		oldProj.Scale(beta)
		addToGrad(proj, oldProj)
	}

	setGrad(grad, x)
}

func (n *NaturalPG) applyFisher(r *RolloutSet, grad anydiff.Grad,
	oldOuts lazyseq.Rereader) anydiff.Grad {
	c := &anyfwd.Creator{
		ValueCreator: creatorFromGrad(grad),
		GradSize:     1,
	}
	fwdBlock, paramMap := n.makeFwd(c, grad)
	fwdIn := &makeFwdTape{Tape: r.Inputs, Creator: c}

	outSeq := &unfwdRereader{
		Fwd:          n.apply(lazyseq.TapeRereader(c, fwdIn), fwdBlock),
		Regular:      oldOuts,
		FwdToRegular: paramMap,
	}
	klSeq := lazyseq.Map(outSeq, func(v anydiff.Res, num int) anydiff.Res {
		zeroGrad := c.ValueCreator.MakeVector(v.Output().Len())
		constVec := v.Output().Copy()
		constVec.(*anyfwd.Vector).Jacobian[0].Set(zeroGrad)
		return n.ActionSpace.KL(anydiff.NewConst(constVec), v, num)
	})
	kl := lazyseq.Mean(klSeq)

	newGrad := anydiff.Grad{}
	for newParam, oldParam := range paramMap {
		if _, ok := grad[oldParam]; ok {
			newGrad[newParam] = c.MakeVector(newParam.Vector.Len())
		}
	}

	one := c.MakeVector(1)
	one.AddScalar(c.MakeNumeric(1))
	kl.Propagate(one, newGrad)

	out := anydiff.Grad{}
	for newParam, paramGrad := range newGrad {
		out[paramMap[newParam]] = paramGrad.(*anyfwd.Vector).Jacobian[0]
	}

	return out
}

func (n *NaturalPG) apply(in lazyseq.Rereader, b anyrnn.Block) lazyseq.Rereader {
	if n.ApplyPolicy == nil {
		cachedIn := lazyseq.Unlazify(in)
		return lazyseq.Lazify(anyrnn.Map(cachedIn, b))
	} else {
		return n.ApplyPolicy(in, b)
	}
}

func (n *NaturalPG) makeFwd(c *anyfwd.Creator, derivs anydiff.Grad) (anyrnn.Block,
	map[*anydiff.Var]*anydiff.Var) {
	fwdBlock, err := serializer.Copy(n.Policy)
	if err != nil {
		panic(err)
	}
	anyfwd.MakeFwd(c, fwdBlock)

	newToOld := map[*anydiff.Var]*anydiff.Var{}
	oldParams := anynet.AllParameters(n.Policy)
	for i, newParam := range anynet.AllParameters(fwdBlock) {
		oldParam := oldParams[i]
		newToOld[newParam] = oldParam
		if deriv, ok := derivs[oldParam]; ok {
			newParam.Vector.(*anyfwd.Vector).Jacobian[0].Set(deriv)
		}
	}

	return fwdBlock.(anyrnn.Block), newToOld
}

func (n *NaturalPG) iters() int {
	if n.Iters != 0 {
		return n.Iters
	} else {
		return DefaultConjGradIters
	}
}

// makeFwdTape wraps a Tape to translate it to a forward
// auto-diff creator.
type makeFwdTape struct {
	Tape    lazyseq.Tape
	Creator *anyfwd.Creator
}

func (m *makeFwdTape) ReadTape(start, end int) <-chan *anyseq.Batch {
	res := make(chan *anyseq.Batch, 1)
	go func() {
		for in := range m.Tape.ReadTape(start, end) {
			newBatch := &anyseq.Batch{
				Present: in.Present,
				Packed:  m.Creator.MakeVector(in.Packed.Len()),
			}
			newBatch.Packed.(*anyfwd.Vector).Values.Set(in.Packed)
			res <- newBatch
		}
		close(res)
	}()
	return res
}

// unfwdRereader is a lazyseq.Rereader used for computing
// Fisher-vector products efficiently.
// When the Fisher-vector product is computed, backprop
// through the network will produce a zero gradient (but
// the gradient has a non-zero derivative).
// This is because the upstream vectors are all zero (with
// non-zero derivatives).
// Thus, we can optimize the back-propagation by avoiding
// forward auto-diff for the backward pass.
type unfwdRereader struct {
	Fwd     lazyseq.Rereader
	Regular lazyseq.Rereader

	FwdToRegular map[*anydiff.Var]*anydiff.Var
}

func (u *unfwdRereader) Creator() anyvec.Creator {
	return u.Fwd.Creator()
}

func (u *unfwdRereader) Forward() <-chan *anyseq.Batch {
	return u.Fwd.Forward()
}

func (u *unfwdRereader) Vars() anydiff.VarSet {
	return u.Fwd.Vars()
}

func (u *unfwdRereader) Reread(start, end int) <-chan *anyseq.Batch {
	return u.Fwd.Reread(start, end)
}

func (u *unfwdRereader) Propagate(upstream <-chan *anyseq.Batch, grad lazyseq.Grad) {
	for _ = range u.Forward() {
	}

	surrogateDownstream := make(chan *anyseq.Batch, 1)
	go func() {
		for in := range upstream {
			surrogateDownstream <- &anyseq.Batch{
				Present: in.Present,
				Packed:  in.Packed.(*anyfwd.Vector).Jacobian[0],
			}
		}
		close(surrogateDownstream)
	}()

	u.Regular.Propagate(surrogateDownstream, &surrogateGrad{
		OrigGrad:     grad,
		FwdToRegular: u.FwdToRegular,
	})
}

type surrogateGrad struct {
	OrigGrad     lazyseq.Grad
	FwdToRegular map[*anydiff.Var]*anydiff.Var
}

func (s *surrogateGrad) Use(f func(g anydiff.Grad)) {
	s.OrigGrad.Use(func(g anydiff.Grad) {
		surrogateGrad := anydiff.Grad{}
		for variable, vec := range g {
			if regularVar, ok := s.FwdToRegular[variable]; !ok {
				panic("superfluous gradient variable")
			} else {
				surrogateGrad[regularVar] = vec.(*anyfwd.Vector).Jacobian[0]
			}
		}
		f(surrogateGrad)
	})
}

func copyGrad(g anydiff.Grad) anydiff.Grad {
	res := anydiff.Grad{}
	for k, v := range g {
		res[k] = v.Copy()
	}
	return res
}

func zeroGrad(g anydiff.Grad) anydiff.Grad {
	res := copyGrad(g)
	res.Clear()
	return res
}

func dotGrad(g1, g2 anydiff.Grad) anyvec.Numeric {
	c := creatorFromGrad(g1)
	ops := c.NumOps()
	sum := c.MakeNumeric(0)
	for variable, grad := range g1 {
		sum = ops.Add(sum, grad.Dot(g2[variable]))
	}
	return sum
}

func addToGrad(dst, src anydiff.Grad) {
	for variable, dstVec := range dst {
		dstVec.Add(src[variable])
	}
}

func subFromGrad(dst, src anydiff.Grad) {
	for variable, dstVec := range dst {
		dstVec.Sub(src[variable])
	}
}

func setGrad(dst, src anydiff.Grad) {
	for variable, dstVec := range dst {
		dstVec.Set(src[variable])
	}
}

func allZeros(grad anydiff.Grad) bool {
	for _, x := range grad {
		sum := anyvec.AbsSum(x)
		zero := x.Creator().MakeNumeric(0)
		ops := x.Creator().NumOps()
		if !ops.Identical(sum, zero) {
			return false
		}
	}
	return true
}
