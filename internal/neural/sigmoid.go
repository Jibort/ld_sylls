// Funció no linial 'Sigmòide' per Lookup table.
// CreatedAt: 2024/11/24 dg. JIQ

package neural

import (
	"fmt"
	"math"
)

const (
	// Mida de la lookup table
	SIGMOID_TABLE_SIZE = 201
)

var (
	// SigmoidTable conté els valors precalculats de la funció sigmoide
	sigmoidTable [SIGMOID_TABLE_SIZE]FixedPoint

	// Marca d'inicialització
	isInitialized bool
)

// Inicialitza la taula sigmòide.
func InitSigmoid() {
	// Inicialitza la lookup table per la funció sigmoide
	for i := 0; i < SIGMOID_TABLE_SIZE; i++ {
		// Convertim l'índex a un valor x entre -1.0 i 1.0
		x := -1.0 + (float64(i) * (2.0 / float64(SIGMOID_TABLE_SIZE-1)))

		// Escalem x per tenir un rang efectiu més ampli de la sigmoide
		scaledX := 4.0 * x

		// Calculem el valor real de la sigmoide
		sigmoidVal := 1.0 / (1.0 + math.Exp(-scaledX))

		// Guardem el resultat com a FixedPoint
		sigmoidTable[i] = FromFloat64(sigmoidVal)

		isInitialized = true
	}
}

// Valida si la taula està inicialitzada.
func CheckInitialized() {
	if !isInitialized {
		InitSigmoid()
	}
}

// Sigmoid retorna el valor de la funció sigmoide utilitzant la lookup table
func Sigmoid(x FixedPoint) FixedPoint {
	CheckInitialized()

	// Convertim el valor x (-1.0 a 1.0) a un índex (0 a 200)
	f := x.ToFloat64()

	// Assegurem que f està dins del rang [-1,1]
	if f <= -1.0 {
		return sigmoidTable[0]
	}
	if f >= 1.0 {
		return sigmoidTable[SIGMOID_TABLE_SIZE-1]
	}

	// Convertim el valor a índex
	// Formula: idx = (x + 1) * (SIZE-1)/2
	idx := int((f + 1.0) * float64(SIGMOID_TABLE_SIZE-1) / 2.0)

	// Per més precisió, podem fer interpolació lineal entre els dos punts més propers
	// Això és opcional però dona resultats més suaus
	if idx < SIGMOID_TABLE_SIZE-1 {
		// Trobem el punt x actual en la nostra escala
		x1 := -1.0 + float64(idx)*2.0/float64(SIGMOID_TABLE_SIZE-1)
		x2 := -1.0 + float64(idx+1)*2.0/float64(SIGMOID_TABLE_SIZE-1)

		// Calculem el factor d'interpolació
		t := (f - x1) / (x2 - x1)

		// Interpolem entre els dos valors
		v1 := sigmoidTable[idx]
		v2 := sigmoidTable[idx+1]

		// La interpolació lineal és: v1 + (v2-v1)*t
		diff := v2.Sub(v1)
		return v1.Add(diff.Mul(FromFloat64(t)))
	}

	return sigmoidTable[idx]
}

// SigmoidDerivative retorna la derivada de la funció sigmoide al punt x
// La derivada de la sigmoide és: sigmoid(x) * (1 - sigmoid(x))
func SigmoidDerivative(x FixedPoint) FixedPoint {
	sx := Sigmoid(x)
	one := FromFloat64(1.0)
	return sx.Mul(one.Sub(sx))
}

// TestSigmoid executa proves bàsiques de la funció sigmoide
func TestSigmoid() {
	// Comprovem alguns valors coneguts
	tests := []struct {
		input    float64
		expected float64
	}{
		{0.0, 0.5},    // Sigmoide de 0 és 0.5
		{-1.0, 0.018}, // Aproximadament per x=-4
		{1.0, 0.982},  // Aproximadament per x=4
		{-0.5, 0.122}, // Valor intermig negatiu
		{0.5, 0.878},  // Valor intermig positiu
	}

	for _, test := range tests {
		input := FromFloat64(test.input)
		result := Sigmoid(input)

		// Comprovem que el resultat està dins d'un marge d'error acceptable
		if math.Abs(result.ToFloat64()-test.expected) > 0.01 {
			panic(fmt.Sprintf("Test fallit per entrada %f: esperat %f, obtingut %f",
				test.input, test.expected, result.ToFloat64()))
		}
	}
}
