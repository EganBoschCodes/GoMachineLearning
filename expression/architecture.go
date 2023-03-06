package expression

type Expression interface {
	GetCache() float32
	Evaluate() float32
	Reset()
}

/*
Constant Values
*/

type Constant struct {
	Value float32
}

func (c *Constant) GetCache() float32 {
	return c.Value
}

func (c *Constant) Evaluate() float32 {
	return c.Value
}

func (c *Constant) Reset() {}

/*
Basic Two-Sided Operators
*/

type Operator struct {
	cache  float32
	cached bool
	left   Expression
	right  Expression
	apply  func(float32, float32) float32
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
}

/*
Single-Input Functions
*/

type ActivationFunction struct {
	cache    float32
	cached   bool
	interior Expression
	apply    func(float32) float32
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
}
