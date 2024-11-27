package main

import (
	nr "github.com/jibort/ld_sylls/internal/neural"
)

func main() {
	inputs, targets, inputSize, outputSize := nr.PrepareCorpus()

	network := nr.InitializeNetwork(inputSize, outputSize)
	nr.TrainModel(network, inputs, targets, 100) // Entrena durant 100 èpoques

	// Dividim les dades per validació (en futur es podria separar millor)
	nr.ValidateModel(network, inputs, targets)
}
