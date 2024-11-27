package neural

import (
	"strings"
)

// ConfigureSyllableSplitter configura una xarxa per separació sil·làbica
func ConfigureSyllableSplitter(maxWordLength int) *Network {
	// Calculem la mida màxima de la sortida (considerant els possibles separadors)
	maxOutputLength := maxWordLength*2 - 1 // Cas pitjor: separador entre cada lletra

	// Definim les mides de les capes
	layerSizes := []int{
		maxWordLength,     // Capa d'entrada (paraula original amb padding)
		maxWordLength * 2, // Primera capa oculta més ampla per detectar patrons
		maxOutputLength,   // Segona capa oculta amb espai pels separadors
		maxOutputLength,   // Capa sortida (paraula amb separadors i padding)
	}

	return NewNetwork(layerSizes)
}

func PrepareCorpus() (rIns [][]FixedPoint, rTgts [][]FixedPoint, rMax_W int, rMax_S int) {
	rMax_W, rMax_S = MaxCorpusLengths()

	for word, sylls := range CorpusTest {
		// Dividim la paraula en síl·labes
		lstSylls := strings.Split(sylls, "_")

		// Preparem les entrades i sortides com a FixedPoint
		input := StringToFixedPoints(word, rMax_W)
		target := PrepareTarget(word, lstSylls, rMax_S)
		rIns = append(rIns, input)
		rTgts = append(rTgts, target)
	}
	return rIns, rTgts, rMax_W, rMax_S
}

// PrepareInput prepara l'entrada de la xarxa
func PrepareInput(pWord string, pMaxLen int) []FixedPoint {
	input := make([]FixedPoint, pMaxLen)

	// Copiem els caràcters de la paraula
	for i, char := range pWord {
		input[i] = FixedPointFromRune(char)
	}

	// Afegim el padding
	for i := len(pWord); i < pMaxLen; i++ {
		input[i] = _padd
	}

	return input
}

// PrepareTarget prepara la sortida esperada
func PrepareTarget(pWord string, pSylls []string, pMaxLength int) []FixedPoint {
	target := make([]FixedPoint, pMaxLength)
	pos := 0

	// Per cada síl·laba excepte l'última
	for _, syll := range pSylls[:len(pSylls)-1] {
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
	lastSyll := pSylls[len(pSylls)-1]
	for _, char := range lastSyll {
		target[pos] = FixedPointFromRune(char)
		pos++
	}

	// Afegim el padding
	for i := pos; i < pMaxLength; i++ {
		target[pos] = _padd
	}

	return target
}

// InterpretOutput converteix la sortida de la xarxa en síl·labes
func InterpretOutput(output []FixedPoint) []string {
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
