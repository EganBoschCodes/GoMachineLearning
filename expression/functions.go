package expression

import (
	"fmt"
	"math"

	"github.com/google/uuid"
)

/*
Classic Sigmoid
*/

type sigmoid struct {
	cache      float32
	cached     bool
	interior   Expression
	derivative Expression
	_uuid      uuid.UUID
}

func (s *sigmoid) Evaluate() float32 {
	if !s.cached {
		s.cache = 1 / (1 + float32(math.Exp(-float64(s.interior.Evaluate()))))

		defer func(s *sigmoid) { s.cached = true }(s)
	}
	return s.cache
}

func (s *sigmoid) Reset() {
	if !s.cached {
		return
	}
	s.cached = false
	s.interior.Reset()
}

func (s *sigmoid) ToString() string {
	return fmt.Sprintf("σ(%s)", s.interior.ToString())
}

func (s *sigmoid) Set(_ float32)            {}
func (s *sigmoid) IsConstant() bool         { return false }
func (s *sigmoid) uuid() uuid.UUID          { return s._uuid }
func (s *sigmoid) Equals(e Expression) bool { return s._uuid == e.uuid() }
func (s *sigmoid) GetPartialDerivative(x Expression) Expression {
	if s.derivative == nil {
		s.derivative = Multiply(s, Subtract(GetConstant(1), s))
	}

	return Multiply(s.interior.GetPartialDerivative(x), s.derivative)
}

/*
ReLu
*/

type relu struct {
	cache    float32
	cached   bool
	interior Expression
	_uuid    uuid.UUID
}

func (r *relu) Evaluate() float32 {
	if !r.cached {
		r.cache = float32(math.Max(0, float64(r.interior.Evaluate())))
		defer func(r *relu) { r.cached = true }(r)
	}
	return r.cache
}

func (r *relu) Reset() {
	if !r.cached {
		return
	}
	r.cached = false
	r.interior.Reset()
}

func (r *relu) ToString() string {
	return fmt.Sprintf("ReLu(%s)", r.interior.ToString())
}

func (r *relu) Set(_ float32)            {}
func (r *relu) IsConstant() bool         { return false }
func (r *relu) uuid() uuid.UUID          { return r._uuid }
func (r *relu) Equals(e Expression) bool { return r._uuid == e.uuid() }
func (r *relu) GetPartialDerivative(x Expression) Expression {
	if r.cache > 0 {
		return r.interior.GetPartialDerivative(x)
	}
	return GetConstant(0)
}

/*
Target Loss Function

Note: Don't use this as a regular difference squared function. Target must be a constant, or the values will be incorrect. This is to save on performance.
*/

type loss struct {
	cache  float32
	cached bool
	target Expression
	value  Expression
	_uuid  uuid.UUID
}

func (l *loss) Evaluate() float32 {
	if !l.cached {
		offset := (l.target.Evaluate() - l.value.Evaluate())
		l.cache = 0.5 * offset * offset
		defer func(l *loss) { l.cached = true }(l)
	}
	return l.cache
}

func (l *loss) Reset() {
	if !l.cached {
		return
	}
	l.cached = false
	l.value.Reset()
}

func (l *loss) ToString() string {
	return fmt.Sprintf("(%s - %s)^2", l.target.ToString(), l.value.ToString())
}

func (l *loss) Set(_ float32)            {}
func (l *loss) IsConstant() bool         { return false }
func (l *loss) uuid() uuid.UUID          { return l._uuid }
func (l *loss) Equals(e Expression) bool { return l._uuid == e.uuid() }
func (l *loss) GetPartialDerivative(x Expression) Expression {
	return Multiply(l.value.GetPartialDerivative(x), Subtract(l.target, l.value))
}
