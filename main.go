package main

import (
	"go-backprop/datasets"
	neuralnetworks "go-backprop/neural-networks"
	"time"
)

func phi(input ...float32) []float32 {
	x := input[0]
	y := input[1]
	return []float32{x, y, x * x, y * y, x * y}
}

func main() {
	network := neuralnetworks.Perceptron{}
	network.Initialize(5, 10, 3)
	network.Phi = phi

	network.LearningRate = 10

	network.Train(datasets.GetSpiralDataset(), 20*time.Second)
}
