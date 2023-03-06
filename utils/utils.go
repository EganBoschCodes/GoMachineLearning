package utils

import "go-backprop/expression"

func Read(expressions []expression.Expression) []float32 {
	output := make([]float32, 0)

	for _, e := range expressions {
		output = append(output, e.Evaluate())
	}

	return output
}
