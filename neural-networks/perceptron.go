package neuralnetworks

import (
	"errors"
	"fmt"
	"go-backprop/datasets"
	"go-backprop/expression"
	"time"
)

type Perceptron struct {
	Input        []expression.Expression
	layers       []Layer
	Output       []expression.Expression
	Target       []expression.Expression
	Loss         expression.Expression
	Phi          func(...float32) []float32
	LearningRate float32
}

func (network *Perceptron) Initialize(inputs int, layerData ...int) {
	network.LearningRate = 1
	network.Phi = defaultPhi

	// Generate references to the input
	network.Input = make([]expression.Expression, 0)
	for i := 0; i < inputs; i++ {
		network.Input = append(network.Input, expression.GetConstant(0))
	}

	// Generate first layer, drawing directly from the input
	network.layers = make([]Layer, 0)
	firstLayer := Layer{}
	firstLayer.Initialize(network.Input, layerData[0])

	network.layers = append(network.layers, firstLayer)

	//Generate all the rest of the layers, drawing from the previous layer
	for i := 1; i < len(layerData); i++ {
		layerInputs := network.layers[i-1].GetOutputs()
		nextLayer := Layer{}
		nextLayer.Initialize(layerInputs, layerData[i])
		network.layers = append(network.layers, nextLayer)
	}

	network.Output = network.layers[len(network.layers)-1].GetOutputs()

	//Generate an Empty Target, and set-up loss function
	network.Target = make([]expression.Expression, 0)
	for i := 0; i < len(network.Output); i++ {
		newTarget := expression.GetConstant(float32(i))
		network.Target = append(network.Target, newTarget)

		network.Loss = expression.Sum(network.Loss, expression.Loss(newTarget, network.Output[i]))
	}

	//Setup Backpropagation For all Neurons
	for i := range network.layers {
		for j := range network.layers[i].Neurons {
			neuron := &network.layers[i].Neurons[j]
			neuron.InitBackprop(network.Loss)
		}
	}
}

func (network *Perceptron) Evaluate(inputs []float32) ([]float32, error) {
	if len(inputs) != len(network.Input) {
		return nil, errors.New(fmt.Sprintf("Inputs passed in are of different length than the input of the network! (%d and %d, respectively).", len(inputs), len(network.Input)))
	}

	for i := 0; i < len(inputs); i++ {
		network.Input[i].Set(inputs[i])
	}

	for _, exp := range network.Output {
		exp.Reset()
	}

	output := make([]float32, 0)
	for _, out := range network.Output {
		output = append(output, out.Evaluate())
	}

	return output, nil
}

func (network *Perceptron) Reset() {
	for _, output := range network.Output {
		output.Reset()
	}
	network.Loss.Reset()
}

func (network *Perceptron) backPropagate() {
	//Calculate the shifts for each neuron
	for _, layer := range network.layers {
		for _, neuron := range layer.Neurons {
			neuron.CalculateShift()
		}
	}

	//Apply all calculated changes
	for _, layer := range network.layers {
		for _, neuron := range layer.Neurons {
			neuron.ApplyShift(network.LearningRate)
		}
	}
}

func (network *Perceptron) setData(datapoint datasets.DataPoint) {
	for i := range network.Input {
		input := network.Input[i]
		input.Set(datapoint.Input[i])
	}
	for i := range network.Target {
		target := network.Target[i]
		target.Set(datapoint.Output[i])
	}
}

func (network *Perceptron) Train(data []datasets.DataPoint, duration time.Duration) {

	if len(data[0].Output) != len(network.Output) {
		fmt.Println("Output does not match the networks shape (You passed", len(data[0].Output), "instead of", len(network.Output), ")")
		return
	}

	for i := range data {
		datapoint := &data[i]
		datapoint.Input = network.Phi(datapoint.Input...)
	}

	if len(data[0].Input) != len(network.Input) {
		fmt.Println("Input does not match the networks shape (You passed", len(data[0].Input), "instead of", len(network.Input), ")")
		return
	}

	currentIndex := 0
	start := time.Now()

	startLoss := float32(0)

	for _, datapoint := range data {
		network.setData(datapoint)
		startLoss += network.Loss.Evaluate()
	}
	startLoss /= float32(len(data))

	iterations := 0
	for time.Since(start) < duration {
		if currentIndex == 0 {
			iterations++
		}
		network.setData(data[currentIndex])

		network.Loss.Evaluate()
		network.backPropagate()
		network.Reset()

		currentIndex = (currentIndex + 1) % len(data)
	}

	finalLoss := float32(0)
	for _, datapoint := range data {
		network.setData(datapoint)
		finalLoss += network.Loss.Evaluate()
	}
	finalLoss /= float32(len(data))

	fmt.Println("Training Finished!\n---------------------\nDataset Size:", len(data), "\nTraining Passes:", iterations, "\nStarting Loss:", startLoss, "\nEnding Loss:", finalLoss)
}

func defaultPhi(inputs ...float32) []float32 {
	return inputs
}
