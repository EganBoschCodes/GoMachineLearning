package main

import (
	"fmt"
	"go-backprop/expression"
)

func main() {
	e := expression.Sum(&expression.Constant{Value: 3.0}, &expression.Constant{Value: 2.0})
	fmt.Println(e.Evaluate())
}
