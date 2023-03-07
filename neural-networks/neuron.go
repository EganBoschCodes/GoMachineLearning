package neuralnetworks

import (
	"go-backprop/expression"
	"math/rand"
)

type Neuron struct {
	Weights  []expression.Expression
	Value    expression.Expression
	Gradient []expression.Expression
	inputs   []expression.Expression
}

func (n *Neuron) InitBackprop(loss expression.Expression) {
	/*sigPrime := expression.Multiply(n.Value, expression.Subtract(&expression.Constant{1}, n.Value))

	shifts := make([]expression.Expression, 0)

	for index, _ := range n.Weights {
		var value expression.Expression
		if index == 0 {
			value = &expression.Constant{1}
		} else {
			value = n.inputs[index - 1]
		}

		shifts = append(shifts, )
	}

	if n.Gradient == nil {
		n.Gradient = make([]expression.Expression, 0)
	}

	n.Gradient = expression.Sum(n.Gradient, gradient)*/
}

func (n *Neuron) Initialize(inputs []expression.Expression) {
	n.inputs = inputs

	n.Weights = make([]expression.Expression, 0)
	for i := 0; i <= len(inputs); i++ {
		weight := expression.GetConstant(float32(rand.NormFloat64()))
		n.Weights = append(n.Weights, weight)
	}

	n.Value = n.Weights[0]
	for i := 1; i <= len(inputs); i++ {
		multiply := expression.Multiply(n.Weights[i], inputs[i-1])
		n.Value = expression.Sum(n.Value, multiply)
	}

	n.Value = expression.Sigmoid(n.Value)
}
