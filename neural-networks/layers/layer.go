package layers

import (
	"go-backprop/expression"
	"go-backprop/neural-networks/neurons"
)

type Layer interface {
	Initialize([]expression.Expression, int)
	GetOutputs() []expression.Expression
	GetNeurons() []neurons.Neuron
	SetActivation(func(expression.Expression) expression.Expression)
}
