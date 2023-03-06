package main

import (
	"fmt"
	"go-backprop/expression"
	neuralnetworks "go-backprop/neural-networks"
)

func main() {
	e := expression.Constant{Value: 3.0}

	neuron := neuralnetworks.Neuron{}
	neuron.Initialize([]expression.Expression{expression.Constant{Value: 1}, expression.Constant{Value: -1}})

	fmt.Println(e.Evaluate())

	fmt.Println(neuron.Value.ToString())
	fmt.Println(neuron.Value.Evaluate())
}
