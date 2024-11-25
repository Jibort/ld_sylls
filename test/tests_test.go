package neural

import (
	"math"
	"testing"

	in "github.com/jibort/ld_sylls/internal"
	nr "github.com/jibort/ld_sylls/internal/neural"
)

// CloseEnough comprova si dos float64 són prou propers
func closeEnough(a, b float64) bool {
	const epsilon = 1e-6 // Definim la tolerància acceptable
	return math.Abs(a-b) < epsilon
}

func TestFixedPoint(t *testing.T) {
	// Test de conversió
	fp := nr.FromFloat64(0.5)
	if !closeEnough(fp.ToFloat64(), 0.5) {
		t.Errorf("Expected approximately 0.5, got %f", fp.ToFloat64())
	}

	// Test d'operacions
	a := nr.FromFloat64(0.3)
	b := nr.FromFloat64(0.2)

	sum := a.Add(b)
	if !closeEnough(sum.ToFloat64(), 0.5) {
		t.Errorf("Expected approximately 0.5, got %f", sum.ToFloat64())
	}

	diff := a.Sub(b)
	if !closeEnough(diff.ToFloat64(), 0.1) {
		t.Errorf("Expected approximately 0.1, got %f", diff.ToFloat64())
	}
}

func TestNetwork(t *testing.T) {
	// Inicialitzem el sistema
	if err := in.InitializeNeural(); err != nil {
		t.Fatalf("Failed to initialize neural network: %v", err)
	}

	// Creem una xarxa simple
	network := nr.NewNetwork([]int{2, 2, 1})

	// Test de forward propagation
	input := []nr.FixedPoint{nr.FromFloat64(0.5), nr.FromFloat64(0.5)}
	output := network.Forward(input)

	if len(output) != 1 {
		t.Errorf("Expected output size 1, got %d", len(output))
	}

	// El valor de sortida hauria d'estar entre 0 i 1
	outVal := output[0].ToFloat64()
	if outVal < 0 || outVal > 1 {
		t.Errorf("Output value out of range: %f", outVal)
	}
}
