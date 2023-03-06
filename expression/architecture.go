package expression

import "fmt"

type Expression interface {
	GetCache() float32
	Evaluate() float32
	Reset()
	ToString() string
}

/*
Constant Values
*/

type Constant struct {
	Value float32
}

func (c Constant) GetCache() float32 {
	return c.Value
}

func (c Constant) Evaluate() float32 {
	return c.Value
}

func (c Constant) Reset() {}

func (c Constant) ToString() string { return fmt.Sprintf("%f", c.Value) }

/*
Basic Two-Sided Operators
*/

type Operator struct {
	cache  float32
	cached bool
	left   Expression
	right  Expression
	apply  func(float32, float32) float32
	name   string
}

func (o *Operator) GetCache() float32 {
	return o.cache
}

func (o *Operator) Evaluate() float32 {
	if !o.cached {
		o.cache = o.apply(o.left.Evaluate(), o.right.Evaluate())
	}
	o.cached = true
	return o.cache
}

func (o *Operator) Reset() {
	o.cached = false
	o.left.Reset()
	o.right.Reset()
}

func (o *Operator) ToString() string {
	return fmt.Sprintf("(%s %s %s)", o.left.ToString(), o.name, o.right.ToString())
}

/*
Single-Input Functions
*/

type ActivationFunction struct {
	cache    float32
	cached   bool
	interior Expression
	apply    func(float32) float32
	name     string
}

func (f *ActivationFunction) GetCache() float32 {
	return f.cache
}

func (f *ActivationFunction) Evaluate() float32 {
	if !f.cached {
		f.cache = f.apply(f.interior.Evaluate())
	}
	f.cached = true
	return f.cache
}

func (f *ActivationFunction) Reset() {
	f.cached = false
	f.interior.Reset()
}

func (f *ActivationFunction) ToString() string {
	return fmt.Sprintf("%s(%s)", f.name, f.interior.ToString())
}
