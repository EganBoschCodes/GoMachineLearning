package neuralnetworks

import "go-backprop/expression"

type Layer struct {
	inputs  []expression.Expression
	neurons []Neuron
}

func (layer *Layer) Initialize(inputs []expression.Expression, size int) {
	layer.inputs = inputs
	layer.neurons = make([]Neuron, 0)
	for i := 0; i < size; i++ {
		neuron := Neuron{}
		neuron.Initialize(inputs)
		layer.neurons = append(layer.neurons, neuron)
	}
}

func (layer *Layer) GetOutputs() []expression.Expression {
	outputs := make([]expression.Expression, 0)
	for _, neuron := range layer.neurons {
		outputs = append(outputs, neuron.Value)
	}
	return outputs
}
