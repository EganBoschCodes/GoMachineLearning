package expression

import "fmt"

type Expression interface {
	GetCache() float32
	Evaluate() float32
	Reset()
	ToString() string
	Set(float32)
}

/*
Constant Values
*/

type Constant struct {
	Value float32
}

func (o *Constant) GetCache() float32 {
	return o.Value
}

func (o *Constant) Evaluate() float32 {
	return o.Value
}

func (o *Constant) Reset() {}

func (o *Constant) ToString() string {
	return fmt.Sprintf("%f", o.Value)
}

func (o *Constant) Set(a float32) { o.Value = a }

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
	if !o.cached {
		fmt.Println("RESETTING ", o.ToString())
	}
	o.cached = false
	o.left.Reset()
	o.right.Reset()
}

func (o *Operator) ToString() string {
	return fmt.Sprintf("(%s %s %s)", o.left.ToString(), o.name, o.right.ToString())
}

func (o *Operator) Set(_ float32) {}

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

	if !f.cached {
		fmt.Println("RESETTING ", f.ToString())
	}
	f.cached = false
	f.interior.Reset()
}

func (f *ActivationFunction) ToString() string {
	return fmt.Sprintf("%s(%s)", f.name, f.interior.ToString())
}

func (f *ActivationFunction) Set(_ float32) {}
