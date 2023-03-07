package main

import (
	"fmt"
	neuralnetworks "go-backprop/neural-networks"
	"go-backprop/utils"
)

func main() {
	network := neuralnetworks.Perceptron{}
	network.Initialize(2, 3, 2)
	network.SetInput([]float32{-1, 1})
	network.LearningRate = 0.1

	for i := 0; i < 5000; i++ {
		output := utils.Read(network.Output)
		loss := network.Loss.Evaluate()
		if i%100 == 0 {
			fmt.Println("Output:", output, ", Loss:", loss)
		}
		network.BackPropagate()
		network.Reset()
	}
}
