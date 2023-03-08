package expression

import (
	"fmt"
	"sync"

	"github.com/google/uuid"
)

type Expression interface {
	Evaluate() float32
	Reset()
	ToString() string
	Set(float32)
	GetPartialDerivative(Expression) Expression
	IsConstant() bool
	Equals(Expression) bool
	uuid() uuid.UUID
}

func GetConstant(val float32) Expression {
	return &constant{Value: val, _uuid: uuid.New()}
}

/*
Constant Values
*/

type constant struct {
	Value float32
	_uuid uuid.UUID
}

func (c *constant) Evaluate() float32        { return c.Value }
func (c *constant) Reset()                   {}
func (c *constant) ToString() string         { return fmt.Sprintf("%f", c.Value) }
func (c *constant) Set(a float32)            { c.Value = a }
func (c *constant) IsConstant() bool         { return true }
func (c *constant) uuid() uuid.UUID          { return c._uuid }
func (c *constant) Equals(e Expression) bool { return c._uuid == e.uuid() }

func (c *constant) GetPartialDerivative(exp Expression) Expression {
	if c.Equals(exp) {
		return GetConstant(1)
	} else {
		return GetConstant(0)
	}
}

/*
Basic Two-Sided Operators
*/

type operator struct {
	cache  float32
	cached bool
	left   Expression
	right  Expression
	apply  func(float32, float32) float32
	derive func(Expression, Expression, Expression) Expression
	name   string
	_uuid  uuid.UUID
	_lock  sync.Mutex
}

func (o *operator) Evaluate() float32 {
	o._lock.Lock()
	defer o._lock.Unlock()
	if !o.cached {
		o.cache = o.apply(o.left.Evaluate(), o.right.Evaluate())
		defer func(o *operator) { o.cached = true }(o)
	}
	return o.cache
}

func (o *operator) Reset() {
	if !o.cached {
		return
	}
	o.cached = false
	o.left.Reset()
	o.right.Reset()
}

func (o *operator) ToString() string {
	return fmt.Sprintf("(%s %s %s)", o.left.ToString(), o.name, o.right.ToString())
}

func (o *operator) Set(_ float32)            {}
func (o *operator) IsConstant() bool         { return false }
func (o *operator) uuid() uuid.UUID          { return o._uuid }
func (o *operator) Equals(e Expression) bool { return o._uuid == e.uuid() }
func (o *operator) GetPartialDerivative(exp Expression) Expression {
	return o.derive(o.left, o.right, exp)
}
