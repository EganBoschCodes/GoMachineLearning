package expression

import (
	"github.com/google/uuid"
)

func Sum(a Expression, b Expression) Expression {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}

	apply := func(a float32, b float32) float32 { return a + b }
	derive := func(l Expression, r Expression, x Expression) Expression {
		drdx := r.GetPartialDerivative(x)
		dldx := l.GetPartialDerivative(x)
		drdxIsZero := drdx.IsConstant() && drdx.Evaluate() == 0
		dldxIsZero := dldx.IsConstant() && dldx.Evaluate() == 0

		if drdxIsZero && dldxIsZero {
			return GetConstant(0)
		} else if drdxIsZero {
			return dldx
		} else if dldxIsZero {
			return drdx
		}

		return Sum(dldx, drdx)
	}
	return &operator{cached: false, left: a, right: b, apply: apply, derive: derive, name: "+", _uuid: uuid.New()}
}

func Subtract(a Expression, b Expression) Expression {
	if a == nil {
		return Multiply(GetConstant(-1), b)
	}
	if b == nil {
		return a
	}

	apply := func(a float32, b float32) float32 { return a - b }
	derive := func(l Expression, r Expression, x Expression) Expression {
		drdx := r.GetPartialDerivative(x)
		dldx := l.GetPartialDerivative(x)
		drdxIsZero := drdx.IsConstant() && drdx.Evaluate() == 0
		dldxIsZero := dldx.IsConstant() && dldx.Evaluate() == 0

		if drdxIsZero && dldxIsZero {
			return GetConstant(0)
		} else if drdxIsZero {
			return dldx
		} else if dldxIsZero {
			return Multiply(GetConstant(-1), drdx)
		}

		return Subtract(dldx, drdx)
	}
	return &operator{cached: false, left: a, right: b, apply: apply, derive: derive, name: "-", _uuid: uuid.New()}
}

func Multiply(a Expression, b Expression) Expression {
	apply := func(a float32, b float32) float32 { return a * b }
	derive := func(l Expression, r Expression, x Expression) Expression {
		drdx := r.GetPartialDerivative(x)
		dldx := l.GetPartialDerivative(x)
		drdxIsZero := drdx.IsConstant() && drdx.Evaluate() == 0
		dldxIsZero := dldx.IsConstant() && dldx.Evaluate() == 0

		if drdxIsZero && dldxIsZero {
			return GetConstant(0)
		} else if drdxIsZero {
			return Multiply(r, dldx)
		} else if dldxIsZero {
			return Multiply(l, drdx)
		}

		return Sum(Multiply(l, drdx), Multiply(r, dldx))
	}
	return &operator{cached: false, left: a, right: b, apply: apply, derive: derive, name: "*", _uuid: uuid.New()}
}

func Loss(target Expression, value Expression) Expression {
	return &loss{target: target, value: value, _uuid: uuid.New()}
}

func Sigmoid(a Expression) Expression {
	return &sigmoid{interior: a, _uuid: uuid.New()}
}

func ReLu(a Expression) Expression {
	return &relu{interior: a, _uuid: uuid.New()}
}
