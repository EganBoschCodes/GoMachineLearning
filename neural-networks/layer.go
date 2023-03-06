package neuralnetworks

import (
	"fmt"
	"go-backprop/expression"
)

type Layer struct {
	inputs  []expression.Expression
	Neurons []Neuron
}

func (layer *Layer) Initialize(inputs []expression.Expression, size int) {

	fmt.Println("NEW LAYER: SIZE", size)
	layer.inputs = inputs
	layer.Neurons = make([]Neuron, 0)
	for i := 0; i < size; i++ {
		neuron := Neuron{}
		neuron.Initialize(inputs)
		layer.Neurons = append(layer.Neurons, neuron)
	}

	//fmt.Println(len(layer.Neurons))

	//for _, neuron := range layer.Neurons {
	//fmt.Println(neuron.Value.ToString())
	//}
}

func (layer *Layer) GetOutputs() []expression.Expression {
	outputs := make([]expression.Expression, 0)
	for _, neuron := range layer.Neurons {
		outputs = append(outputs, neuron.Value)
	}
	return outputs
}
