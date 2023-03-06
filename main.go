package main

import (
	"fmt"
	neuralnetworks "go-backprop/neural-networks"
	"go-backprop/utils"
)

func main() {

	network := neuralnetworks.Perceptron{}

	network.Initialize(2, 1)
	network.Layers[0].Neurons[0].Value.Reset()

	fmt.Println(utils.Read(network.Output))

	network.SetInput([]float32{-1, 1})
	network.Layers[0].Neurons[0].Value.Reset()

	fmt.Println(utils.Read(network.Output))

}
