package main

import (
	"fmt"
	"go-backprop/expression"
)

func main() {

	var left expression.Expression = expression.GetConstant(2)
	var right expression.Expression = expression.GetConstant(-1)

	multiple := expression.Multiply(left, right)
	sigmoid := expression.Sigmoid(multiple)

	fmt.Println("Sigmoid Value:", sigmoid.Evaluate())
	fmt.Println("Sigmoid Formula:", sigmoid.ToString())

	sigleft := sigmoid.GetPartialDerivative(left)
	fmt.Println("ds/dl Formula:", sigleft.ToString())
	fmt.Println("ds/dl Value:", sigleft.Evaluate())
	sigright := sigmoid.GetPartialDerivative(right)
	fmt.Println("ds/dr Formula:", sigright.ToString())
	fmt.Println("ds/dr Value:", sigright.Evaluate())

}
