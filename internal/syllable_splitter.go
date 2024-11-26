package internal

import (
	nr "github.com/jibort/ld_sylls/internal/neural"
)

var _padd = FixedPointFromRune('~')
var _ssep = FixedPointFromRune('_')

// ConfigureSyllableSplitter configura una xarxa per separació sil·làbica
func ConfigureSyllableSplitter(maxWordLength int) *nr.Network {
	// Calculem la mida màxima de la sortida (considerant els possibles separadors)
	maxOutputLength := maxWordLength*2 - 1 // Cas pitjor: separador entre cada lletra

	// Definim les mides de les capes
	layerSizes := []int{
		maxWordLength,     // Capa d'entrada (paraula original amb padding)
		maxWordLength * 2, // Primera capa oculta més ampla per detectar patrons
		maxOutputLength,   // Segona capa oculta amb espai pels separadors
		maxOutputLength,   // Capa sortida (paraula amb separadors i padding)
	}

	return nr.NewNetwork(layerSizes)
}

// PrepareInput prepara l'entrada de la xarxa
func PrepareInput(word string, maxLength int) []nr.FixedPoint {
	input := make([]nr.FixedPoint, maxLength)

	// Copiem els caràcters de la paraula
	for i, char := range word {
		input[i] = FixedPointFromRune(char)
	}

	// Afegim el padding
	for i := len(word); i < maxLength; i++ {
		input[i] = _padd
	}

	return input
}

// PrepareTarget prepara la sortida esperada
func PrepareTarget(word string, syllables []string, maxLength int) []nr.FixedPoint {
	target := make([]nr.FixedPoint, maxLength)
	pos := 0

	// Per cada síl·laba excepte l'última
	for _, syll := range syllables[:len(syllables)-1] {
		// Copiem els caràcters de la síl·laba
		for _, char := range syll {
			target[pos] = FixedPointFromRune(char)
			pos++
		}
		// Afegim el separador després de la síl·laba
		target[pos] = _ssep
		pos++
	}

	// Última síl·laba (sense separador al final)
	lastSyll := syllables[len(syllables)-1]
	for _, char := range lastSyll {
		target[pos] = FixedPointFromRune(char)
		pos++
	}

	// Afegim el padding
	for i := pos; i < maxLength; i++ {
		target[pos] = _padd
	}

	return target
}

// InterpretOutput converteix la sortida de la xarxa en síl·labes
func InterpretOutput(output []nr.FixedPoint) []string {
	var syllables []string
	var currentSyll string

	for _, fp := range output {
		char := RuneFromFixedPoint(fp)

		switch char {
		case '~': // Final de la seqüència
			if currentSyll != "" {
				syllables = append(syllables, currentSyll)
			}
			return syllables
		case '_': // Separador de síl·labes
			if currentSyll != "" {
				syllables = append(syllables, currentSyll)
				currentSyll = ""
			}
		default: // Caràcter normal
			currentSyll += string(char)
		}
	}

	// Si queda alguna síl·laba sense processar
	if currentSyll != "" {
		syllables = append(syllables, currentSyll)
	}

	return syllables
}

// // Exemple d'ús
// func ExampleUsage() {
// 	maxWordLength := 32
// 	network := ConfigureSyllableSplitter(maxWordLength)

// 	// Exemple d'entrenament
// 	word := "casa"
// 	expectedSyllables := []string{"ca", "sa"}

// 	// Preparar input i target
// 	input := PrepareInput(word, maxWordLength)
// 	target := PrepareTarget(word, expectedSyllables, maxWordLength*2-1)

// 	// Forward i backward (durant l'entrenament)
// 	output := network.Forward(input)
// 	// network.Backward(...) // Amb els gradients corresponents

// 	// Interpretar resultat
// 	predictedSyllables := InterpretOutput(output)
// }
