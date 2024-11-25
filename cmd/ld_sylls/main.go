package main

import (
	"fmt"
	"log"

	in "github.com/jibort/ld_sylls/internal"
	"github.com/jibort/ld_sylls/internal/neural"
)

func main() {
	// Inicialitzem el sistema neural
	if err := in.InitializeNeural(); err != nil {
		log.Fatalf("Error initializing neural network: %v", err)
	}

	// Creem una xarxa d'exemple
	// Input: 10 neurones (per exemple, per a una finestra de 10 caràcters)
	// Hidden: 8 neurones
	// Output: 2 neurones (probabilitat de tall de síl·laba o no)
	network := neural.NewNetwork([]int{10, 8, 2})

	// Exemple d'ús [codi de prova]
	input := make([]neural.FixedPoint, 10)
	for i := range input {
		input[i] = neural.FromFloat64(0.1) // Valors d'exemple
	}

	output := network.Forward(input)
	fmt.Printf("Output: %v\n", output)
}
