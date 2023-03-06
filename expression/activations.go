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
		s.cache = 1 / (1 + float32(math.Exp(float64(s.interior.Evaluate()))))
	} else {
		fmt.Println("Reading Cache: ", s.ToString())
	}
	s.cached = true
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
	} else {
		fmt.Println("Reading Cache: ", r.ToString())
	}
	r.cached = true
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
