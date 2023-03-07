package main

import (
	"fmt"
	neuralnetworks "go-backprop/neural-networks"
	"go-backprop/utils"
)

func main() {
	network := neuralnetworks.Perceptron{}
	network.Initialize(2, 1)
	firstInput := network.Input[0]
	network.SetInput([]float32{-1, 1})

	fmt.Println(firstInput.ToString())

	partialDerivative := network.Output[0].GetPartialDerivative(firstInput)
	for i := 0; i < 5000; i++ {
		output := utils.Read(network.Output)
		loss := network.Loss.Evaluate()
		pd := partialDerivative.Evaluate()

		fmt.Println("Output:", output, ", Loss:", loss, ", PD:", pd)
		firstInput.Set(firstInput.Evaluate() - pd)

		network.Reset()
		partialDerivative.Reset()
	}
	fmt.Println(firstInput.ToString())
}
