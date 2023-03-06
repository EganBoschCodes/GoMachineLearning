package main

import (
	"fmt"
	"go-backprop/expression"
)

func main() {

	internal := expression.Constant{-1}
	var test expression.Expression = internal
	expr := expression.Sigmoid(&test)

	fmt.Println(expr.ToString())

	test.Set(2)
	fmt.Println(test.ToString())

	fmt.Println(expr.ToString())

}
