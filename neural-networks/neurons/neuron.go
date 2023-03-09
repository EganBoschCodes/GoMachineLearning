package neurons

import (
	"go-backprop/expression"
)

type Neuron interface {
	InitBackprop(expression.Expression)
	Initialize([]expression.Expression)
	CalculateShift()
	ApplyShift(float32)
	GetValue() expression.Expression
}
