package main

import (
	"go-backprop/datasets"
	neuralnetworks "go-backprop/neural-networks/networks"
	"time"
)

func phi(input ...float32) []float32 {
	x := input[0]
	y := input[1]
	return []float32{x, y, x * y, y * y, x * x}
}

func main() {
	// Create network with 4 inputs, 7 hidden neurons, and 3 outputs
	network := neuralnetworks.Perceptron{}
	network.Initialize(4, 7, 3)

	// Default: 1
	network.LearningRate = 0.5

	// Some ways you can modify input datasets
	spiral := datasets.GetSpiralDataset()
	datasets.ApplyPhi(spiral, phi)
	datasets.PCAReduce(spiral, 4)

	// Specify the dataset and the time to train
	network.Train(spiral, 5*time.Second)
}
