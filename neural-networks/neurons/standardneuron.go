package neurons

import (
	"go-backprop/expression"
	"math/rand"
)

type StandardNeuron struct {
	weights    []expression.Expression
	value      expression.Expression
	gradient   []expression.Expression
	inputs     []expression.Expression
	shift      []float32
	Index      int
	Layer      int
	Activation func(expression.Expression) expression.Expression
}

func (n *StandardNeuron) InitBackprop(loss expression.Expression) {
	n.gradient = make([]expression.Expression, 0)
	n.shift = make([]float32, 0)
	for _, weight := range n.weights {
		n.gradient = append(n.gradient, loss.GetPartialDerivative(weight))
		n.shift = append(n.shift, 0)
	}
}

func (n *StandardNeuron) Initialize(inputs []expression.Expression) {
	n.inputs = inputs

	n.weights = make([]expression.Expression, 0)
	for i := 0; i <= len(inputs); i++ {
		weight := expression.GetConstant(float32(rand.NormFloat64()))
		n.weights = append(n.weights, weight)
	}

	n.value = n.weights[0]
	for i := 1; i <= len(inputs); i++ {
		multiply := expression.Multiply(n.weights[i], inputs[i-1])
		n.value = expression.Sum(n.value, multiply)
	}

	n.value = n.Activation(n.value)
}

func (n *StandardNeuron) CalculateShift() {
	for index, gradient := range n.gradient {
		n.shift[index] = gradient.Evaluate()
	}
}

func (n *StandardNeuron) ApplyShift(learningRate float32) {
	//fmt.Println("Read     ", n.layer, n.index, n.shift)
	for index, shift := range n.shift {
		n.weights[index].Set(n.weights[index].Evaluate() + learningRate*shift)
		n.gradient[index].Reset()
	}
}

func (n *StandardNeuron) GetValue() expression.Expression {
	return n.value
}
