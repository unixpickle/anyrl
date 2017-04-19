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

	// FwdDiff copies the Policy and changes it to use an
	// anyfwd.Creator with the derivatives given in g.
	// Any gradients missing from g should be set to 0.
	//
	// Since the resulting block will have a different set
	// of parameter pointers, this returns a mapping from
	// the new parameters to old parameters.
	//
	// If nil, the package-level MakeFwdDiff is used.
	FwdDiff func(c *anyfwd.Creator, p anyrnn.Block, g anydiff.Grad) (anyrnn.Block,
		map[*anydiff.Var]*anydiff.Var)

	// ApplyPolicy applies a policy to an input sequence.
	// If nil, back-propagation through time is used.
	ApplyPolicy func(s lazyseq.Rereader, b anyrnn.Block) lazyseq.Rereader
}

// Run computes the natural gradient for the rollouts.
func (n *NaturalPG) Run(r *RolloutSet) anydiff.Grad {
	grad := anydiff.NewGrad(n.Params...)
	if len(grad) == 0 {
		return grad
	}

	PolicyGradient(n.ActionSpace, r, grad, func(in lazyseq.Rereader) lazyseq.Rereader {
		return n.apply(in, n.Policy)
	})

	// TODO: add option for running CG on a subset of the
	// total experience.

	n.conjugateGradients(r, grad)

	return grad
}

func (n *NaturalPG) conjugateGradients(r *RolloutSet, grad anydiff.Grad) {
	c := creatorFromGrad(grad)
	storedOuts := n.storePolicyOutputs(c, r)

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
		appliedProj := n.applyFisher(r, proj, storedOuts)

		// (r dot r) / (p dot A*p)
		alpha := quotient(c, residualMag, dotGrad(proj, appliedProj))

		// x = x + alpha*p
		alphaProj := copyGrad(proj)
		alphaProj.Scale(alpha)
		addToGrad(x, alphaProj)

		// r = r - alpha*A*p
		appliedProj.Scale(alpha)
		subFromGrad(residual, appliedProj)

		// (newR dot newR) / (r dot r)
		newResidualMag := dotGrad(residual, residual)
		beta := quotient(c, newResidualMag, residualMag)
		residualMag = newResidualMag

		// p = beta*p + r
		oldProj := proj
		proj = copyGrad(residual)
		oldProj.Scale(beta)
		addToGrad(proj, oldProj)
	}

	setGrad(grad, x)
}

func (n *NaturalPG) storePolicyOutputs(c anyvec.Creator, r *RolloutSet) lazyseq.Tape {
	tape, writer := lazyseq.ReferenceTape()

	out := n.apply(lazyseq.TapeRereader(c, r.Inputs), n.Policy)
	for outVec := range out.Forward() {
		writer <- &anyseq.Batch{
			Present: outVec.Present,
			Packed:  outVec.Packed.Copy(),
		}
	}

	close(writer)
	return tape
}

func (n *NaturalPG) applyFisher(r *RolloutSet, grad anydiff.Grad,
	oldOuts lazyseq.Tape) anydiff.Grad {
	c := &anyfwd.Creator{
		ValueCreator: creatorFromGrad(grad),
		GradSize:     1,
	}
	fwdOldOuts := &makeFwdTape{Tape: oldOuts, Creator: c}
	fwdBlock, paramMap := n.makeFwd(c, grad)
	fwdIn := &makeFwdTape{Tape: r.Inputs, Creator: c}

	outSeq := &unfwdRereader{
		Fwd:          n.apply(lazyseq.TapeRereader(c, fwdIn), fwdBlock),
		Regular:      n.apply(lazyseq.TapeRereader(c, r.Inputs), n.Policy),
		FwdToRegular: paramMap,
	}
	klDiv := lazyseq.Mean(lazyseq.MapN(func(num int, v ...anydiff.Res) anydiff.Res {
		return n.ActionSpace.KL(v[0], v[1], num)
	}, lazyseq.TapeRereader(c, fwdOldOuts), outSeq))

	newGrad := anydiff.Grad{}
	for newParam, oldParam := range paramMap {
		if _, ok := grad[oldParam]; ok {
			newGrad[newParam] = c.MakeVector(newParam.Vector.Len())
		}
	}

	one := c.MakeVector(1)
	one.AddScalar(c.MakeNumeric(1))
	klDiv.Propagate(one, newGrad)

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

func (n *NaturalPG) makeFwd(c *anyfwd.Creator, g anydiff.Grad) (anyrnn.Block,
	map[*anydiff.Var]*anydiff.Var) {
	if n.FwdDiff == nil {
		return MakeFwdDiff(c, n.Policy, g)
	} else {
		return n.FwdDiff(c, n.Policy, g)
	}
}

func (n *NaturalPG) iters() int {
	if n.Iters != 0 {
		return n.Iters
	} else {
		return DefaultConjGradIters
	}
}

// MakeFwdDiff copies the RNN policy, updates it to use
// forward automatic differentiation, and sets the forward
// derivatives to the vectors in g.
// It returns the new block and a mapping from the new
// parameters to the old ones.
//
// Copying is done by serializing the policy and then
// deserializing it once more.
// If the policy does not implement serializer.Serializer,
// this will fail.
//
// Setting the creator is done by looping through the
// policy parameters and updating all of their vectors.
// If there are hidden parameter vectors, this will fail.
func MakeFwdDiff(c *anyfwd.Creator, p anyrnn.Block, g anydiff.Grad) (anyrnn.Block,
	map[*anydiff.Var]*anydiff.Var) {
	data, err := serializer.SerializeAny(p)
	if err != nil {
		panic(err)
	}
	var newPolicy anyrnn.Block
	if err := serializer.DeserializeAny(data, &newPolicy); err != nil {
		panic(err)
	}

	oldParams := anynet.AllParameters(p)
	newToOld := map[*anydiff.Var]*anydiff.Var{}
	for i, param := range anynet.AllParameters(newPolicy) {
		oldParam := oldParams[i]
		newToOld[param] = oldParam
		newVec := c.MakeVector(param.Vector.Len()).(*anyfwd.Vector)
		newVec.Values.Set(param.Vector)
		if gradVec, ok := g[oldParam]; ok {
			newVec.Jacobian[0].Set(gradVec)
		}
		param.Vector = newVec
	}

	return newPolicy, newToOld
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
	sum := c.MakeVector(1)
	for variable, grad := range g1 {
		sum.AddScalar(grad.Dot(g2[variable]))
	}
	return anyvec.Sum(sum)
}

func quotient(c anyvec.Creator, num, denom anyvec.Numeric) anyvec.Numeric {
	vec := c.MakeVector(1)
	vec.AddScalar(denom)
	anyvec.Pow(vec, c.MakeNumeric(-1))
	vec.Scale(num)
	return anyvec.Sum(vec)
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
