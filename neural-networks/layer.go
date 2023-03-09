package neuralnetworks

import (
	"go-backprop/expression"
)

type Layer interface {
	Initialize([]expression.Expression, int)
	GetOutputs() []expression.Expression
	GetNeurons() []Neuron
	SetActivation(func(expression.Expression) expression.Expression)
}

/*
STANDARD LAYER - Multiply all inputs by weight, add bias, pass into an activation function
*/

type StandardLayer struct {
	inputs     []expression.Expression
	neurons    []Neuron
	index      int
	activation func(expression.Expression) expression.Expression
}

func (layer *StandardLayer) Initialize(inputs []expression.Expression, size int) {
	layer.inputs = inputs
	layer.neurons = make([]Neuron, 0)
	for i := 0; i < size; i++ {
		neuron := &standardNeuron{index: i, layer: layer.index, activation: layer.activation}
		neuron.Initialize(inputs)
		layer.neurons = append(layer.neurons, neuron)
	}
}

func (layer *StandardLayer) GetOutputs() []expression.Expression {
	outputs := make([]expression.Expression, 0)
	for _, neuron := range layer.neurons {
		outputs = append(outputs, neuron.GetValue())
	}
	return outputs
}

func (layer *StandardLayer) GetNeurons() []Neuron {
	return layer.neurons
}

func (layer *StandardLayer) SetActivation(activation func(expression.Expression) expression.Expression) {
	layer.activation = activation
}

/*
BATCHNORM LAYER - Just take in the normalized inputs and multiply by a std dev and add a new mean
*/

type BatchNormLayer struct {
	inputs  []expression.Expression
	neurons []Neuron
	index   int
}

func (layer *BatchNormLayer) Initialize(inputs []expression.Expression, size int) {
	layer.inputs = inputs
	layer.neurons = make([]Neuron, 0)
	for i := 0; i < size; i++ {
		neuron := &batchnormNeuron{index: i, layer: layer.index}
		neuron.Initialize(inputs)
		layer.neurons = append(layer.neurons, neuron)
	}
}

func (layer *BatchNormLayer) GetOutputs() []expression.Expression {
	outputs := make([]expression.Expression, 0)
	for _, neuron := range layer.neurons {
		outputs = append(outputs, neuron.GetValue())
	}
	return outputs
}

func (layer *BatchNormLayer) GetNeurons() []Neuron {
	return layer.neurons
}

func (layer *BatchNormLayer) SetActivation(activation func(expression.Expression) expression.Expression) {
}
