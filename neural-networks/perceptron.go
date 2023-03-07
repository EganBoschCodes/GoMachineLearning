package neuralnetworks

import (
	"errors"
	"fmt"
	"go-backprop/expression"
)

type Perceptron struct {
	Input  []expression.Expression
	Layers []Layer
	Output []expression.Expression
	Target []expression.Expression
	Loss   expression.Expression
}

func (network *Perceptron) Initialize(inputs int, layerData ...int) {
	// Generate references to the input
	network.Input = make([]expression.Expression, 0)
	for i := 0; i < inputs; i++ {
		network.Input = append(network.Input, expression.GetConstant(0))
	}

	// Generate first layer, drawing directly from the input
	network.Layers = make([]Layer, 0)
	firstLayer := Layer{}
	firstLayer.Initialize(network.Input, layerData[0])

	network.Layers = append(network.Layers, firstLayer)

	//Generate all the rest of the layers, drawing from the previous layer
	for i := 1; i < len(layerData); i++ {
		layerInputs := network.Layers[i-1].GetOutputs()
		nextLayer := Layer{}
		nextLayer.Initialize(layerInputs, layerData[i])
		network.Layers = append(network.Layers, nextLayer)
	}

	network.Output = network.Layers[len(network.Layers)-1].GetOutputs()

	//Generate an Empty Target, and set-up loss function
	network.Target = make([]expression.Expression, 0)
	for i := 0; i < len(network.Output); i++ {
		newTarget := expression.GetConstant(0)
		network.Target = append(network.Target, newTarget)

		network.Loss = expression.Sum(network.Loss, expression.Loss(newTarget, network.Output[i]))
	}

	//Setup Backpropagation For all Neurons

}

func (network *Perceptron) SetInput(inputs []float32) error {
	if len(inputs) != len(network.Input) {
		return errors.New(fmt.Sprintf("Inputs passed in are of different length than the input of the network! (%d and %d, respectively).", len(inputs), len(network.Input)))
	}

	for i := 0; i < len(inputs); i++ {
		network.Input[i].Set(inputs[i])
	}

	for _, exp := range network.Output {
		exp.Reset()
	}

	return nil
}

func (network *Perceptron) Reset() {
	for _, output := range network.Output {
		output.Reset()
	}
	network.Loss.Reset()
}
