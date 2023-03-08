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
	shift    []float32
	index    int
	layer    int
}

func (n *Neuron) InitBackprop(loss expression.Expression) Neuron {
	n.Gradient = make([]expression.Expression, 0)
	n.shift = make([]float32, 0)
	for _, weight := range n.Weights {
		n.Gradient = append(n.Gradient, loss.GetPartialDerivative(weight))
		n.shift = append(n.shift, 0)
	}
	return *n
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

func (n *Neuron) CalculateShift() {
	for index, gradient := range n.Gradient {
		n.shift[index] = gradient.Evaluate()
	}
}

func (n *Neuron) ApplyShift(learningRate float32) {
	//fmt.Println("Read     ", n.layer, n.index, n.shift)
	for index, shift := range n.shift {
		n.Weights[index].Set(n.Weights[index].Evaluate() + learningRate*shift)
		n.Gradient[index].Reset()
	}
}
