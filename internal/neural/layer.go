// Estructures de Capes i Xarxes.
// CreatedAt: 2024/11/24 dg. JIQ

package neural

import "fmt"

// Layer representa una capa de la xarxa neuronal
type Layer struct {
	// Nombre de neurones d'entrada i sortida
	InputSize  int
	OutputSize int

	// Pesos i biaixos utilitzant el nostre tipus FixedPoint
	Weights [][]FixedPoint
	Biases  []FixedPoint

	// Valors intermedis per backpropagation
	LastInput  []FixedPoint
	LastOutput []FixedPoint

	WeightGradients [][]FixedPoint // Gradients dels pesos
	BiasGradients   []FixedPoint   // Gradients dels biaixos
	InputGradients  []FixedPoint   // Gradients per la capa anterior
}

// NewLayer crea una nova capa amb pesos aleatoris inicialitzats
func NewLayer(inputSize, outputSize int) Layer {
	layer := Layer{
		InputSize:       inputSize,
		OutputSize:      outputSize,
		Weights:         make([][]FixedPoint, outputSize),
		Biases:          make([]FixedPoint, outputSize),
		LastInput:       make([]FixedPoint, inputSize),
		LastOutput:      make([]FixedPoint, outputSize),
		WeightGradients: make([][]FixedPoint, outputSize),
		BiasGradients:   make([]FixedPoint, outputSize),
		InputGradients:  make([]FixedPoint, inputSize),
	}

	// Inicialitzem les matrius de gradients
	for i := range layer.WeightGradients {
		layer.WeightGradients[i] = make([]FixedPoint, inputSize)
	}

	// Inicialitza els pesos amb valors petits aleatoris
	for i := range layer.Weights {
		layer.Weights[i] = make([]FixedPoint, inputSize)
		for j := range layer.Weights[i] {
			// Inicialitzem amb valors entre -0.1 i 0.1
			layer.Weights[i][j] = RandomFixedPoint(-0.1, 0.1)
		}
	}

	// Inicialitza els biaixos a zero
	for i := range layer.Biases {
		layer.Biases[i] = FromFloat64(0.0)
	}

	return layer
}

// Forward realitza la propagació cap endavant per una capa
func (l *Layer) Forward(input []FixedPoint) []FixedPoint {
	if len(input) != l.InputSize {
		panic("Mida d'entrada incorrecta")
	}

	// Guarda l'entrada per backpropagation
	copy(l.LastInput, input)

	// Calcula la sortida per cada neurona
	for i := 0; i < l.OutputSize; i++ {
		sum := l.Biases[i]
		for j, val := range input {
			sum = sum.Add(l.Weights[i][j].Mul(val))
		}
		l.LastOutput[i] = Sigmoid(sum)
	}

	return l.LastOutput
}

// ResetGradients posa a zero tots els gradients per començar una nova època
func (l *Layer) ResetGradients() {
	for i := range l.LastInput {
		l.LastInput[i] = FromFloat64(0.0)
	}
	for i := range l.LastOutput {
		l.LastOutput[i] = FromFloat64(0.0)
	}
}

// Clone crea una còpia profunda de la capa
func (l *Layer) Clone() Layer {
	clone := Layer{
		InputSize:  l.InputSize,
		OutputSize: l.OutputSize,
		Weights:    make([][]FixedPoint, l.OutputSize),
		Biases:     make([]FixedPoint, l.OutputSize),
		LastInput:  make([]FixedPoint, l.InputSize),
		LastOutput: make([]FixedPoint, l.OutputSize),
	}

	// Copiem els pesos
	for i := range l.Weights {
		clone.Weights[i] = make([]FixedPoint, l.InputSize)
		copy(clone.Weights[i], l.Weights[i])
	}

	// Copiem els biaixos i els últims valors
	copy(clone.Biases, l.Biases)
	copy(clone.LastInput, l.LastInput)
	copy(clone.LastOutput, l.LastOutput)

	return clone
}

// String retorna una representació en string de la capa
func (l *Layer) String() string {
	return fmt.Sprintf("Layer[%d→%d]", l.InputSize, l.OutputSize)
}
