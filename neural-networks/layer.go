package neuralnetworks

import (
	"go-backprop/expression"
)

type Layer struct {
	inputs  []expression.Expression
	Neurons []Neuron
	index   int
}

func (layer *Layer) Initialize(inputs []expression.Expression, size int) {
	layer.inputs = inputs
	layer.Neurons = make([]Neuron, 0)
	for i := 0; i < size; i++ {
		neuron := Neuron{index: i, layer: layer.index}
		neuron.Initialize(inputs)
		layer.Neurons = append(layer.Neurons, neuron)
	}
}

func (layer *Layer) GetOutputs() []expression.Expression {
	outputs := make([]expression.Expression, 0)
	for _, neuron := range layer.Neurons {
		outputs = append(outputs, neuron.Value)
	}
	return outputs
}
