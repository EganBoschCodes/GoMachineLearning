package layers

import (
	"go-backprop/expression"
	"go-backprop/neural-networks/neurons"
)

type StandardLayer struct {
	inputs     []expression.Expression
	neurons    []neurons.Neuron
	Index      int
	activation func(expression.Expression) expression.Expression
}

func (layer *StandardLayer) Initialize(inputs []expression.Expression, size int) {
	layer.inputs = inputs
	layer.neurons = make([]neurons.Neuron, 0)
	for i := 0; i < size; i++ {
		neuron := &neurons.StandardNeuron{Index: i, Layer: layer.Index, Activation: layer.activation}
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

func (layer *StandardLayer) GetNeurons() []neurons.Neuron {
	return layer.neurons
}

func (layer *StandardLayer) SetActivation(activation func(expression.Expression) expression.Expression) {
	layer.activation = activation
}
