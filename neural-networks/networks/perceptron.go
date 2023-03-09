package neuralnetworks

import (
	"errors"
	"fmt"
	"go-backprop/datasets"
	"go-backprop/expression"
	"go-backprop/neural-networks/layers"
	"go-backprop/neural-networks/neurons"
	"sync"
	"time"
)

type Perceptron struct {
	Input        []expression.Expression
	Layers       []layers.Layer
	Output       []expression.Expression
	Target       []expression.Expression
	Loss         expression.Expression
	Phi          func(...float32) []float32
	LearningRate float32
	numNeurons   int
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
	network.Layers = make([]layers.Layer, 0)
	batchNormLayer := &layers.BatchNormLayer{Index: 0}
	batchNormLayer.Initialize(network.Input, len(network.Input))
	network.numNeurons = len(network.Input)

	network.Layers = append(network.Layers, batchNormLayer)

	//Generate all the rest of the Layers, drawing from the previous layer
	for i := 0; i < len(layerData); i++ {
		layerInputs := network.Layers[i].GetOutputs()
		nextLayer := &layers.StandardLayer{Index: i + 1}
		nextLayer.SetActivation(expression.Sigmoid)
		nextLayer.Initialize(layerInputs, layerData[i])
		network.Layers = append(network.Layers, nextLayer)

		network.numNeurons += layerData[i]
	}

	network.Output = network.Layers[len(network.Layers)-1].GetOutputs()

	//Generate an Empty Target, and set-up loss function
	network.Target = make([]expression.Expression, 0)
	for i := 0; i < len(network.Output); i++ {
		newTarget := expression.GetConstant(float32(i))
		network.Target = append(network.Target, newTarget)

		network.Loss = expression.Sum(network.Loss, expression.Loss(newTarget, network.Output[i]))
	}

	//Setup Backpropagation For all Neurons
	for i := range network.Layers {
		neurons := network.Layers[i].GetNeurons()
		for j := range neurons {
			neuron := neurons[j]
			neuron.InitBackprop(network.Loss)
		}
	}
}

func (network *Perceptron) Evaluate(rawinputs []float32) ([]float32, error) {
	inputs := network.Phi(rawinputs...)

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
	// Calculate the backpropagation shifts in a multithreaded fashion
	func() {
		var waitGroup sync.WaitGroup
		waitGroup.Add(network.numNeurons)
		defer waitGroup.Wait()

		//Calculate the shifts for each neuron
		for i := len(network.Layers) - 1; i >= 0; i-- {
			layer := network.Layers[i]
			for _, neuron := range layer.GetNeurons() {
				go func(neuron neurons.Neuron) {
					defer waitGroup.Done()
					neuron.CalculateShift()
				}(neuron)
			}
		}
	}()

	//Apply all calculated changes
	for _, layer := range network.Layers {
		for _, neuron := range layer.GetNeurons() {
			neuron.ApplyShift(network.LearningRate)
		}
	}
}

func (network *Perceptron) setData(datapoint datasets.DataPoint) {
	network.Loss.Reset()

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

	datasets.NormalizeInputs(data)

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

	for i := range data {
		datapoint := data[i]
		network.setData(datapoint)
		startLoss += network.Loss.Evaluate()
		network.Loss.Reset()
	}
	startLoss /= float32(len(data))

	iterations := 0
	totalPoints := 0
	for time.Since(start) < duration {
		if currentIndex == 0 {
			iterations++
		}
		network.setData(data[currentIndex])

		network.backPropagate()
		network.Reset()

		currentIndex = (currentIndex + 1) % len(data)
		totalPoints++
	}

	finalLoss := float32(0)
	for _, datapoint := range data {
		network.setData(datapoint)
		finalLoss += network.Loss.Evaluate()
		network.Loss.Reset()
	}
	finalLoss /= float32(len(data))

	fmt.Println("Training Finished!\n---------------------\nDataset Size:", len(data), "\nTraining Passes:", iterations, "\nTotal Iterations:", totalPoints, "\nStarting Loss:", startLoss, "\nEnding Loss:", finalLoss)
}

func defaultPhi(inputs ...float32) []float32 {
	return inputs
}
