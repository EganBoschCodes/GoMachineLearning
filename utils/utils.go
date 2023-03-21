package utils

import (
	"go-backprop/expression"
	"math"
)

func Read(expressions []expression.Expression) []float32 {
	output := make([]float32, 0)

	for _, e := range expressions {
		output = append(output, e.Evaluate())
	}

	return output
}

func Min(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

func GetDistribution(values []float32) (float32, float32) {
	mean := float32(0)
	for _, val := range values {
		mean += val
	}
	mean /= float32(len(values))

	variance := float32(0)
	for _, val := range values {
		variance += (val - mean) * (val - mean)
	}
	variance /= float32(len(values))

	return mean, float32(math.Sqrt(float64(variance)))
}

func Dot(a []float32, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	sum := float32(0)
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func Float32(a []float64) []float32 {
	ret := make([]float32, len(a))
	for i := range a {
		ret[i] = float32(a[i])
	}
	return ret
}
