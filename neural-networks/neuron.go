package neuralnetworks

import (
	"go-backprop/expression"
	"math/rand"
)

type Neuron interface {
	InitBackprop(expression.Expression)
	Initialize([]expression.Expression)
	CalculateShift()
	ApplyShift(float32)
	GetValue() expression.Expression
}

type standardNeuron struct {
	weights    []expression.Expression
	value      expression.Expression
	gradient   []expression.Expression
	inputs     []expression.Expression
	shift      []float32
	index      int
	layer      int
	activation func(expression.Expression) expression.Expression
}

func (n *standardNeuron) InitBackprop(loss expression.Expression) {
	n.gradient = make([]expression.Expression, 0)
	n.shift = make([]float32, 0)
	for _, weight := range n.weights {
		n.gradient = append(n.gradient, loss.GetPartialDerivative(weight))
		n.shift = append(n.shift, 0)
	}
}

func (n *standardNeuron) Initialize(inputs []expression.Expression) {
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

	n.value = n.activation(n.value)
}

func (n *standardNeuron) CalculateShift() {
	for index, gradient := range n.gradient {
		n.shift[index] = gradient.Evaluate()
	}
}

func (n *standardNeuron) ApplyShift(learningRate float32) {
	//fmt.Println("Read     ", n.layer, n.index, n.shift)
	for index, shift := range n.shift {
		n.weights[index].Set(n.weights[index].Evaluate() + learningRate*shift)
		n.gradient[index].Reset()
	}
}

func (n *standardNeuron) GetValue() expression.Expression {
	return n.value
}

type batchnormNeuron struct {
	weights  []expression.Expression
	value    expression.Expression
	gradient []expression.Expression
	input    expression.Expression
	shift    []float32
	index    int
	layer    int
}

func (n *batchnormNeuron) InitBackprop(loss expression.Expression) {
	n.gradient = make([]expression.Expression, 0)
	n.shift = make([]float32, 0)
	for _, weight := range n.weights {
		n.gradient = append(n.gradient, loss.GetPartialDerivative(weight))
		n.shift = append(n.shift, 0)
	}
}

func (n *batchnormNeuron) Initialize(inputs []expression.Expression) {
	n.input = inputs[n.index]

	n.weights = []expression.Expression{expression.GetConstant(0.000001), expression.GetConstant(1)}
	n.value = expression.Sum(n.weights[0], expression.Multiply(n.weights[1], n.input))
}

func (n *batchnormNeuron) CalculateShift() {
	for index, gradient := range n.gradient {
		n.shift[index] = gradient.Evaluate()
	}
}

func (n *batchnormNeuron) ApplyShift(learningRate float32) {
	for index, shift := range n.shift {
		n.weights[index].Set(n.weights[index].Evaluate() + learningRate*shift)
		//fmt.Println(n.layer, n.index, n.weights[index].Evaluate())
		n.gradient[index].Reset()
	}
}

func (n *batchnormNeuron) GetValue() expression.Expression {
	return n.value
}
