package layers

import (
	"go-backprop/expression"
	"go-backprop/neural-networks/neurons"
)

type BatchNormLayer struct {
	inputs  []expression.Expression
	neurons []neurons.Neuron
	Index   int
}

func (layer *BatchNormLayer) Initialize(inputs []expression.Expression, size int) {
	layer.inputs = inputs
	layer.neurons = make([]neurons.Neuron, 0)
	for i := 0; i < size; i++ {
		neuron := &neurons.BatchnormNeuron{Index: i, Layer: layer.Index}
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

func (layer *BatchNormLayer) GetNeurons() []neurons.Neuron {
	return layer.neurons
}

func (layer *BatchNormLayer) SetActivation(activation func(expression.Expression) expression.Expression) {
}
