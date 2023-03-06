package neuralnetworks

import (
	"go-backprop/expression"
	"math/rand"
)

type Neuron struct {
	Weights  []expression.Constant
	Value    expression.Expression
	Gradient expression.Expression
}

func (n *Neuron) Backprop(gradient expression.Expression) {
	if n.Gradient == nil {
		n.Gradient = gradient
		return
	}

	n.Gradient = expression.Sum(n.Gradient, gradient)
}

func (n *Neuron) Initialize(inputs []expression.Expression) {
	n.Weights = make([]expression.Constant, 0)
	for i := 0; i <= len(inputs); i++ {
		weight := expression.Constant{Value: float32(rand.NormFloat64())}
		n.Weights = append(n.Weights, weight)
	}

	n.Value = n.Weights[0]
	for i := 1; i <= len(inputs); i++ {
		n.Value = expression.Sum(n.Value, expression.Multiply(n.Weights[i], inputs[i-1]))
	}

	n.Value = expression.Sigmoid(n.Value)
}
