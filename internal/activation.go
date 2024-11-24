package internal

import (
	"sync"

	nlf "github.com/jibort/ld_sylls/internal/neural"
)

var (
	initOnce      sync.Once
	isInitialized bool
)

// InitializeNeural inicialitza tots els components necessaris per la xarxa neuronal
func InitializeNeural() error {
	if isInitialized {
		return nil
	}

	initOnce.Do(func() {
		nlf.InitSigmoid()
		isInitialized = true
	})

	return nil
}

func checkInitialized() {
	if !isInitialized {
		panic("Neural network system not initialized. Call InitializeNeural() first")
	}
}
