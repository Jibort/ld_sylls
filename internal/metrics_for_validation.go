package internal

// import (
// 	"fmt"

// 	nr "github.com/jibort/ld_sylls/internal/neural"
// )

// // ValidationResults conté totes les mètriques de validació
// type ValidationResults struct {
// 	// Mètriques generals
// 	TotalWords       int
// 	CorrectWords     int
// 	TotalSyllables   int
// 	CorrectSyllables int

// 	// Mètriques específiques
// 	WordAccuracy     nr.FixedPoint // Proporció de paraules correctes
// 	SyllableAccuracy nr.FixedPoint // Proporció de síl·labes correctes

// 	// Errors específics catalans
// 	DiphthongErrors int // Errors en diftongs (ex: "aigua" → "a-i-gua")
// 	HiatusErrors    int // Errors en hiats (ex: "veïna" → "vei-na")
// 	StressErrors    int // Errors en vocals tòniques

// 	// Matriu de confusió per tipus d'error
// 	ErrorMatrix map[string]int
// }

// // NewValidationResults crea una nova instància de resultats
// func NewValidationResults() *ValidationResults {
// 	return &ValidationResults{
// 		ErrorMatrix: make(map[string]int),
// 	}
// }

// // ValidateNetwork avalua la xarxa amb un conjunt de test
// func ValidateNetwork(network *nr.Network, testData map[string]string, maxWordLength int) *ValidationResults {
// 	results := NewValidationResults()

// 	for word, expectedSyllables := range testData {
// 		// Preparem input
// 		input := PrepareInput(word, maxWordLength)

// 		// Obtenim predicció
// 		output := network.Forward(input)
// 		predictedSyllables := InterpretOutput(output)

// 		// Comparem amb expected
// 		results.evaluateWord(word, expectedSyllables, predictedSyllables)
// 	}

// 	// Calculem mètriques finals
// 	results.calculateMetrics()

// 	return results
// }

// // evaluateWord avalua una predicció individual
// func (vr *ValidationResults) evaluateWord(word, expected string, predicted []string) {
// 	vr.TotalWords++
// 	expectedSylls := splitSyllables(expected)

// 	// Comprovem si la paraula sencera és correcta
// 	if syllablesEqual(expectedSylls, predicted) {
// 		vr.CorrectWords++
// 	}

// 	// Avaluem cada síl·laba
// 	vr.evaluateSyllables(expectedSylls, predicted)

// 	// Analitzem errors específics
// 	vr.analyzeSpecificErrors(word, expectedSylls, predicted)
// }

// // evaluateSyllables avalua les síl·labes individualment
// func (vr *ValidationResults) evaluateSyllables(expected, predicted []string) {
// 	vr.TotalSyllables += len(expected)

// 	// Comptem síl·labes correctes utilitzant LCS (Longest Common Subsequence)
// 	correct := longestCommonSyllables(expected, predicted)
// 	vr.CorrectSyllables += correct
// }

// // analyzeSpecificErrors analitza tipus d'errors específics del català
// func (vr *ValidationResults) analyzeSpecificErrors(word string, expected, predicted []string) {
// 	// Analitzem diftongs
// 	if containsDiphthong(word) && !correctDiphthongSplit(expected, predicted) {
// 		vr.DiphthongErrors++
// 		vr.ErrorMatrix["diphthong"]++
// 	}

// 	// Analitzem hiats
// 	if containsHiatus(word) && !correctHiatusSplit(expected, predicted) {
// 		vr.HiatusErrors++
// 		vr.ErrorMatrix["hiatus"]++
// 	}

// 	// Analitzem vocals tòniques
// 	if hasStressedVowel(word) && !correctStressSplit(expected, predicted) {
// 		vr.StressErrors++
// 		vr.ErrorMatrix["stress"]++
// 	}
// }

// // calculateMetrics calcula les mètriques finals
// func (vr *ValidationResults) calculateMetrics() {
// 	if vr.TotalWords > 0 {
// 		vr.WordAccuracy = nr.FromFloat64(float64(vr.CorrectWords) / float64(vr.TotalWords))
// 	}
// 	if vr.TotalSyllables > 0 {
// 		vr.SyllableAccuracy = nr.FromFloat64(float64(vr.CorrectSyllables) / float64(vr.TotalSyllables))
// 	}
// }

// // Funcions auxiliars

// func containsDiphthong(word string) bool {
// 	diphthongs := []string{"ai", "ei", "oi", "ui", "au", "eu", "iu", "ou"}
// 	for _, d := range diphthongs {
// 		if contains(word, d) {
// 			return true
// 		}
// 	}
// 	return false
// }

// func containsHiatus(word string) bool {
// 	// Cerca de vocals amb dièresi o combinacions específiques
// 	return contains(word, "ï") || contains(word, "ü") ||
// 		contains(word, "ia") || contains(word, "ie") || contains(word, "io") || contains(word, "iu")
// }

// func hasStressedVowel(word string) bool {
// 	stressedVowels := []rune{'à', 'è', 'é', 'í', 'ò', 'ó', 'ú'}
// 	for _, c := range word {
// 		for _, v := range stressedVowels {
// 			if c == v {
// 				return true
// 			}
// 		}
// 	}
// 	return false
// }

// // String genera un informe formatat dels resultats
// func (vr *ValidationResults) String() string {
// 	return fmt.Sprintf(
// 		"Resultats de validació:\n"+
// 			"Total paraules: %d\n"+
// 			"Paraules correctes: %d (%.2f%%)\n"+
// 			"Total síl·labes: %d\n"+
// 			"Síl·labes correctes: %d (%.2f%%)\n"+
// 			"Errors en diftongs: %d\n"+
// 			"Errors en hiats: %d\n"+
// 			"Errors en vocals tòniques: %d\n",
// 		vr.TotalWords,
// 		vr.CorrectWords,
// 		vr.WordAccuracy.ToFloat64()*100,
// 		vr.TotalSyllables,
// 		vr.CorrectSyllables,
// 		vr.SyllableAccuracy.ToFloat64()*100,
// 		vr.DiphthongErrors,
// 		vr.HiatusErrors,
// 		vr.StressErrors,
// 	)
// }
