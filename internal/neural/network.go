// Estructures de Capes i Xarxes.
// CreatedAt: 2024/11/24 dg. JIQ

package neural

import (
	"fmt"
	"math/rand/v2"
)

// Network representa la xarxa neuronal completa
type Network struct {
	Layers []Layer

	// Paràmetres d'entrenament
	LearningRate FixedPoint
	LastError    FixedPoint // Per guardar l'últim error calculat
}

func InitializeNetwork(pInSize, pOutSize int) *Network {
	layerSizes := []int{pInSize, pInSize, pOutSize, pOutSize} // Configuració bàsica de capes
	return NewNetwork(layerSizes)
}

// NewNetwork crea una nova xarxa neuronal
func NewNetwork(layerSizes []int) *Network {
	if len(layerSizes) < 2 {
		panic("La xarxa ha de tenir almenys una capa d'entrada i una de sortida")
	}

	network := &Network{
		Layers:       make([]Layer, len(layerSizes)-1),
		LearningRate: FromFloat64(0.01),
	}

	for i := 0; i < len(layerSizes)-1; i++ {
		network.Layers[i] = NewLayer(layerSizes[i], layerSizes[i+1])
	}

	return network
}

// Forward realitza la propagació cap endavant per tota la xarxa
func (n *Network) Forward(input []FixedPoint) []FixedPoint {
	currentOutput := input

	// Propaga a través de cada capa
	for i := range n.Layers {
		currentOutput = n.Layers[i].Forward(currentOutput)
	}

	return currentOutput
}

// RandomFixedPoint genera un número aleatori FixedPoint entre min i max
func RandomFixedPoint(min, max float64) FixedPoint {
	random := min + rand.Float64()*(max-min)
	return FromFloat64(random)
}

func TrainModel(network *Network, inputs, targets [][]FixedPoint, epochs int) {
	fmt.Println("Començant l'entrenament...")
	network.Train(inputs, targets, epochs) // Entrenament complet
	fmt.Println("Entrenament finalitzat.")
}

// Train entrena la xarxa amb un conjunt de dades
// Actualitzem la funció Train per utilitzar backpropagation
func (n *Network) Train(inputs [][]FixedPoint, targets [][]FixedPoint, epochs int) {
	if len(inputs) != len(targets) {
		panic("El número d'entrades i sortides esperades no coincideix")
	}

	for epoch := 0; epoch < epochs; epoch++ {
		epochError := FromFloat64(0.0)

		for i := range inputs {
			// Forward pass
			output := n.Forward(inputs[i])

			// Calculem l'error
			error := n.calculateError(output, targets[i])
			epochError = epochError.Add(error)

			// Backpropagation
			n.Backpropagate(targets[i])
		}

		// Guardem l'error mitjà d'aquesta època
		n.LastError = epochError.Mul(FromFloat64(1.0 / float64(len(inputs))))
	}
}

// calculateError calcula l'error quadràtic mitjà
func (n *Network) calculateError(output, target []FixedPoint) FixedPoint {
	if len(output) != len(target) {
		panic("Les dimensions de sortida i target no coincideixen")
	}

	error := FromFloat64(0.0)
	for i := range output {
		diff := output[i].Sub(target[i])
		error = error.Add(diff.Mul(diff))
	}

	return error.Mul(FromFloat64(0.5))
}

// Predict fa una predicció amb l'entrada donada
func (n *Network) Predict(input []FixedPoint) []FixedPoint {
	return n.Forward(input)
}

// Clone crea una còpia independent de la xarxa
func (n *Network) Clone() *Network {
	clone := &Network{
		Layers:       make([]Layer, len(n.Layers)),
		LearningRate: n.LearningRate,
		LastError:    n.LastError,
	}

	for i := range n.Layers {
		clone.Layers[i] = n.Layers[i].Clone()
	}

	return clone
}

// String retorna una representació en string de la xarxa
func (n *Network) String() string {
	s := "Neural Network:\n"
	for i, layer := range n.Layers {
		s += fmt.Sprintf("  Layer %d: %s\n", i, layer.String())
	}
	return s
}

// SetLearningRate permet modificar el learning rate
func (n *Network) SetLearningRate(rate float64) {
	if rate <= 0 {
		panic("Learning rate ha de ser positiu")
	}
	n.LearningRate = FromFloat64(rate)
}

// GetLastError retorna l'últim error calculat
func (n *Network) GetLastError() float64 {
	return n.LastError.ToFloat64()
}

func (n *Network) Backpropagate(target []FixedPoint) {
	outputLayer := &n.Layers[len(n.Layers)-1]
	if len(target) != outputLayer.OutputSize {
		panic("La mida del target no coincideix amb la sortida de la xarxa")
	}

	// 1. Calculem els gradients de la capa de sortida
	n.backpropOutputLayer(target)

	// 2. Propaguem els gradients per les capes ocultes
	for i := len(n.Layers) - 2; i >= 0; i-- {
		n.backpropHiddenLayer(i)
	}

	// 3. Actualitzem els pesos i biaixos de totes les capes
	n.updateWeights()
}

// backpropOutputLayer calcula els gradients per la capa de sortida
func (n *Network) backpropOutputLayer(target []FixedPoint) {
	layer := &n.Layers[len(n.Layers)-1]

	// Per cada neurona de sortida
	for i := 0; i < layer.OutputSize; i++ {
		// Calculem l'error: (output - target)
		output := layer.LastOutput[i]
		error := output.Sub(target[i])

		// Calculem la derivada de la sigmoide
		sigmoidDeriv := SigmoidDerivative(layer.LastOutput[i])

		// El gradient de sortida és error * sigmoid'(output)
		outputGradient := error.Mul(sigmoidDeriv)

		// Calculem els gradients dels pesos i biaixos
		for j := range layer.WeightGradients[i] {
			// gradient = outputGradient * input
			layer.WeightGradients[i][j] = outputGradient.Mul(layer.LastInput[j])
		}

		// El gradient del biaix és simplement el gradient de sortida
		layer.BiasGradients[i] = outputGradient

		// Guardem el gradient per la capa anterior
		for j := range layer.InputGradients {
			layer.InputGradients[j] = layer.Weights[i][j].Mul(outputGradient)
		}
	}
}

// backpropHiddenLayer calcula els gradients per una capa oculta
func (n *Network) backpropHiddenLayer(layerIndex int) {
	layer := &n.Layers[layerIndex]
	nextLayer := &n.Layers[layerIndex+1]

	// Per cada neurona en aquesta capa
	for i := 0; i < layer.OutputSize; i++ {
		// Calculem el gradient que ve de la capa següent
		sumGradient := FromFloat64(0.0)
		for j := 0; j < nextLayer.OutputSize; j++ {
			nextGradient := nextLayer.InputGradients[i]
			sumGradient = sumGradient.Add(nextGradient)
		}

		// Calculem la derivada de la sigmoide
		sigmoidDeriv := SigmoidDerivative(layer.LastOutput[i])

		// El gradient final per aquesta neurona
		hiddenGradient := sumGradient.Mul(sigmoidDeriv)

		// Actualitzem els gradients dels pesos i biaixos
		for j := range layer.WeightGradients[i] {
			layer.WeightGradients[i][j] = hiddenGradient.Mul(layer.LastInput[j])
		}

		layer.BiasGradients[i] = hiddenGradient

		// Propaguem els gradients a la capa anterior
		for j := range layer.InputGradients {
			layer.InputGradients[j] = layer.Weights[i][j].Mul(hiddenGradient)
		}
	}
}

// updateWeights actualitza els pesos i biaixos utilitzant els gradients calculats
func (n *Network) updateWeights() {
	for l := range n.Layers {
		layer := &n.Layers[l]

		// Actualitzem els pesos
		for i := range layer.Weights {
			for j := range layer.Weights[i] {
				// weight = weight - learning_rate * gradient
				update := n.LearningRate.Mul(layer.WeightGradients[i][j])
				layer.Weights[i][j] = layer.Weights[i][j].Sub(update)
			}
		}

		// Actualitzem els biaixos
		for i := range layer.Biases {
			update := n.LearningRate.Mul(layer.BiasGradients[i])
			layer.Biases[i] = layer.Biases[i].Sub(update)
		}
	}
}

func ValidateModel(network *Network, validationInputs, validationTargets [][]FixedPoint) {
	correct := 0
	for i, input := range validationInputs {
		fmt.Printf("valida: '%s' -> ", FixedPointsToString(input))
		predicted := network.Predict(input)
		fmt.Printf("'%s'\n", FixedPointsToString(predicted))
		if IsPredictionCorrect(predicted, validationTargets[i]) {
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(validationInputs)) * 100
	fmt.Printf("Precisió de validació: %.2f%%\n", accuracy)
}

func IsPredictionCorrect(predicted, target []FixedPoint) bool {
	// Compara la predicció i el target (implementació simplificada)
	ok := len(predicted) == len(target) && fmt.Sprintf("%v", predicted) == fmt.Sprintf("%v", target)
	fmt.Printf("'%s' == '%s': %v\n", FixedPointsToString(predicted), FixedPointsToString(target), ok)
	return ok
}
