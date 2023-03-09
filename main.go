package main

import (
	"go-backprop/datasets"
	neuralnetworks "go-backprop/neural-networks/networks"
	"time"
)

func phi(input ...float32) []float32 {
	x := input[0]
	y := input[1]
	return []float32{x, y, x * x, y * y, x * y}
}

func main() {
	// Create network with 5 inputs, 7 hidden neurons, and 3 outputs
	network := neuralnetworks.Perceptron{}
	network.Initialize(5, 7, 3)

	// Optional: If you don't define this, it will just take the data-point input as passed
	network.Phi = phi

	// Default: 1
	network.LearningRate = 0.5

	network.Train(datasets.GetSpiralDataset(), 30*time.Second)
}
