package neuralnetworks

import (
	"fmt"
	"go-backprop/expression"
)

type Layer struct {
	inputs  []expression.Expression
	neurons []Neuron
}

func (layer *Layer) Initialize(inputs *[]expression.Expression, size int) {

	fmt.Println("NEW LAYER: SIZE", size)
	layer.inputs = *inputs
	layer.neurons = make([]Neuron, 0)
	for i := 0; i < size; i++ {
		neuron := Neuron{}
		neuron.Initialize(inputs)
		layer.neurons = append(layer.neurons, neuron)
	}

	//fmt.Println(len(layer.neurons))

	//for _, neuron := range layer.neurons {
	//fmt.Println(neuron.Value.ToString())
	//}
}

func (layer *Layer) GetOutputs() *[]expression.Expression {
	outputs := make([]expression.Expression, 0)
	for _, neuron := range layer.neurons {
		outputs = append(outputs, neuron.Value)
	}
	return &outputs
}
