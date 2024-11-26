// Taula de símbols i la seva correspondència amb valors dins el rang -1.0 a +1.0.
// CreatedAt: 2024/11/25 dl. JIQ

package internal

import (
	nr "github.com/jibort/ld_sylls/internal/neural"
)

// Més freqüents sonores: B, D, G, L, M, N, R
// Més freqüents sordes:  C, F, P, S, T, X
// Menys feqüents: 		  J, Q, V, Z, ·
// Habituals estrangeres: H, K, W, Y
var Symbols = map[rune]nr.FixedPoint{
	'_': nr.FromFloat64(+1.000),

	',': nr.FromFloat64(+0.990),
	';': nr.FromFloat64(+0.980),
	'(': nr.FromFloat64(+0.970),
	'[': nr.FromFloat64(+0.960),
	'{': nr.FromFloat64(-0.950),
	'"': nr.FromFloat64(+0.940),
	'¿': nr.FromFloat64(+0.930),
	'¡': nr.FromFloat64(+0.920),

	'ï': nr.FromFloat64(+0.855),

	'0': nr.FromFloat64(+0.840),
	'1': nr.FromFloat64(+0.835),
	'2': nr.FromFloat64(+0.830),
	'3': nr.FromFloat64(+0.825),
	'4': nr.FromFloat64(+0.820),

	'b': nr.FromFloat64(+0.805),
	'd': nr.FromFloat64(+0.755),
	'g': nr.FromFloat64(+0.705),

	'c': nr.FromFloat64(+0.665),
	'ç': nr.FromFloat64(+0.635),
	'f': nr.FromFloat64(+0.605),
	'p': nr.FromFloat64(+0.555),

	'a': nr.FromFloat64(+0.505),
	'à': nr.FromFloat64(+0.405),

	'j': nr.FromFloat64(+0.375),
	'q': nr.FromFloat64(+0.325),

	'e': nr.FromFloat64(+0.305),
	'è': nr.FromFloat64(+0.205),
	'é': nr.FromFloat64(+0.105),

	'h': nr.FromFloat64(+0.075),
	'k': nr.FromFloat64(+0.025),
	'i': nr.FromFloat64(+0.005),

	'~': nr.FromFloat64(0.000), // Símbol pels inputs finals després dels runes del mot.

	'í': nr.FromFloat64(-0.005),

	'w': nr.FromFloat64(-0.025),
	'y': nr.FromFloat64(-0.075),

	'o': nr.FromFloat64(-0.105),
	'ò': nr.FromFloat64(-0.205),
	'ó': nr.FromFloat64(-0.305),

	'v': nr.FromFloat64(-0.375),
	'z': nr.FromFloat64(-0.325),

	'u': nr.FromFloat64(-0.405),
	'·': nr.FromFloat64(-0.425),
	'ú': nr.FromFloat64(-0.505),

	'x': nr.FromFloat64(-0.555),
	't': nr.FromFloat64(-0.605),
	's': nr.FromFloat64(-0.665),

	'r': nr.FromFloat64(-0.705),
	'n': nr.FromFloat64(-0.755),
	'm': nr.FromFloat64(-0.805),

	'5': nr.FromFloat64(-0.920),
	'6': nr.FromFloat64(+0.925),
	'7': nr.FromFloat64(-0.930),
	'8': nr.FromFloat64(-0.935),
	'9': nr.FromFloat64(-0.940),

	'l': nr.FromFloat64(-0.855),

	'!':  nr.FromFloat64(-0.920),
	'?':  nr.FromFloat64(-0.930),
	'\'': nr.FromFloat64(-0.940),
	'}':  nr.FromFloat64(-0.950),
	']':  nr.FromFloat64(-0.960),
	')':  nr.FromFloat64(+0.970),
	':':  nr.FromFloat64(-0.980),
	'.':  nr.FromFloat64(-0.990),

	'¬': nr.FromFloat64(-1.000), // Rune per a símbol desconegut.
}

func StringToFixedPoints(pStr string) (rFPs []nr.FixedPoint) {
	for r := range pStr {
		fp, exists := Symbols[rune(r)]
		if !exists {
			fp = Symbols['¬']
		}
		rFPs = append(rFPs, fp)
	}

	return
}

func FixedPointFromRune(pChar rune) (rFp nr.FixedPoint) {
	rFp = Symbols['¬']
	for rn, fp := range Symbols {
		if rn == pChar {
			rFp = fp
			break
		}
	}
	return
}

func RuneFromFixedPoint(pFP nr.FixedPoint) (rRune rune) {
	rRune = '¬'
	for rn, fp := range Symbols {
		if fp.Equal(pFP) {
			rRune = rn
			break
		}
	}
	return
}
