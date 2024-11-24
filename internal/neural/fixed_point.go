// Representació d'un número decimal en el rang -1.0 fins a +1.0
// disposant d'un bit pel signe i 31 bits per a la part fraccionària.
// Això correspon a uns 9 dígits decimals significatius.
// CreatedAt: 2024/11/24 dg. JIQ

package neural

import (
	"fmt"
	"math"
)

// FixedPoint representa un número decimal entre -1.0 i +1.0
type FixedPoint int32

const (
	// Definim les constants pel tipus
	SCALE   = int32(math.MaxInt32) // 2^31 - 1
	MAX_VAL = FixedPoint(SCALE)
	MIN_VAL = FixedPoint(-SCALE)
)

// FromFloat64 converteix un float64 a FixedPoint
func FromFloat64(f float64) FixedPoint {
	if f >= 1.0 {
		return MAX_VAL
	}
	if f <= -1.0 {
		return MIN_VAL
	}
	return FixedPoint(f * float64(SCALE))
}

// ToFloat64 converteix FixedPoint a float64
func (fp FixedPoint) ToFloat64() float64 {
	return float64(fp) / float64(SCALE)
}

// Add suma dos FixedPoint, amb saturació
func (fp FixedPoint) Add(other FixedPoint) FixedPoint {
	sum := int64(fp) + int64(other)
	if sum > int64(MAX_VAL) {
		return MAX_VAL
	}
	if sum < int64(MIN_VAL) {
		return MIN_VAL
	}
	return FixedPoint(sum)
}

// Sub resta dos FixedPoint, amb saturació
func (fp FixedPoint) Sub(other FixedPoint) FixedPoint {
	diff := int64(fp) - int64(other)
	if diff > int64(MAX_VAL) {
		return MAX_VAL
	}
	if diff < int64(MIN_VAL) {
		return MIN_VAL
	}
	return FixedPoint(diff)
}

// Mul multiplica dos FixedPoint, amb saturació
func (fp FixedPoint) Mul(other FixedPoint) FixedPoint {
	// Utilitzem int64 per evitar overflow en la multiplicació
	prod := (int64(fp) * int64(other)) / int64(SCALE)
	if prod > int64(MAX_VAL) {
		return MAX_VAL
	}
	if prod < int64(MIN_VAL) {
		return MIN_VAL
	}
	return FixedPoint(prod)
}

// LessThan retorna true si el valor actual és menor que other
func (fp FixedPoint) LessThan(other FixedPoint) bool {
	return int32(fp) < int32(other)
}

// GreaterThan retorna true si el valor actual és major que other
func (fp FixedPoint) GreaterThan(other FixedPoint) bool {
	return int32(fp) > int32(other)
}

// LessOrEqual retorna true si el valor actual és menor o igual que other
func (fp FixedPoint) LessOrEqual(other FixedPoint) bool {
	return int32(fp) <= int32(other)
}

// GreaterOrEqual retorna true si el valor actual és major o igual que other
func (fp FixedPoint) GreaterOrEqual(other FixedPoint) bool {
	return int32(fp) >= int32(other)
}

// Equal retorna true si el valor actual és igual a other
func (fp FixedPoint) Equal(other FixedPoint) bool {
	return int32(fp) == int32(other)
}

// String implementa l'interface Stringer
func (fp FixedPoint) String() string {
	return fmt.Sprintf("%f", fp.ToFloat64())
}
