package neurons

import "go-backprop/expression"

type BatchnormNeuron struct {
	weights  []expression.Expression
	value    expression.Expression
	gradient []expression.Expression
	input    expression.Expression
	shift    []float32
	Index    int
	Layer    int
}

func (n *BatchnormNeuron) InitBackprop(loss expression.Expression) {
	n.gradient = make([]expression.Expression, 0)
	n.shift = make([]float32, 0)
	for _, weight := range n.weights {
		n.gradient = append(n.gradient, loss.GetPartialDerivative(weight))
		n.shift = append(n.shift, 0)
	}
}

func (n *BatchnormNeuron) Initialize(inputs []expression.Expression) {
	n.input = inputs[n.Index]

	n.weights = []expression.Expression{expression.GetConstant(0.000001), expression.GetConstant(1)}
	n.value = expression.Sum(n.weights[0], expression.Multiply(n.weights[1], n.input))
}

func (n *BatchnormNeuron) CalculateShift() {
	for index, gradient := range n.gradient {
		n.shift[index] = gradient.Evaluate()
	}
}

func (n *BatchnormNeuron) ApplyShift(learningRate float32) {
	for index, shift := range n.shift {
		n.weights[index].Set(n.weights[index].Evaluate() + learningRate*shift)
		//fmt.Println(n.layer, n.index, n.weights[index].Evaluate())
		n.gradient[index].Reset()
	}
}

func (n *BatchnormNeuron) GetValue() expression.Expression {
	return n.value
}
