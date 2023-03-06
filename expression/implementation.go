package expression

import "math"

func Sum(a Expression, b Expression) Expression {
	return &Operator{cached: false, left: a, right: b, apply: func(a float32, b float32) float32 { return a + b }, name: "+"}
}

func Multiply(a Expression, b Expression) Expression {
	return &Operator{cached: false, left: a, right: b, apply: func(a float32, b float32) float32 { return a * b }, name: "*"}
}

func Sigmoid(a Expression) Expression {
	return &ActivationFunction{cached: false, interior: a, apply: func(a float32) float32 { return 1 / (1 + float32(math.Exp(-float64(a)))) }, name: "σ"}
}
