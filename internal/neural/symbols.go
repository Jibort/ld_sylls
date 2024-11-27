// Taula de símbols i la seva correspondència amb valors dins el rang -1.0 a +1.0.
// CreatedAt: 2024/11/25 dl. JIQ

package neural

import (
	"strings"
)

var _padd = FixedPointFromRune('~')
var _ssep = FixedPointFromRune('_')
var _errc = FixedPointFromRune('¬')

// Més freqüents sonores: B, D, G, L, M, N, R
// Més freqüents sordes:  C, F, P, S, T, X
// Menys feqüents: 		  J, Q, V, Z, ·
// Habituals estrangeres: H, K, W, Y
var Symbols = map[rune]FixedPoint{
	'_': FromFloat64(+1.000),

	',': FromFloat64(+0.990),
	';': FromFloat64(+0.980),
	'(': FromFloat64(+0.970),
	'[': FromFloat64(+0.960),
	'{': FromFloat64(-0.950),
	'"': FromFloat64(+0.940),
	'¿': FromFloat64(+0.930),
	'¡': FromFloat64(+0.920),

	'ï': FromFloat64(+0.855),

	'0': FromFloat64(+0.840),
	'1': FromFloat64(+0.835),
	'2': FromFloat64(+0.830),
	'3': FromFloat64(+0.825),
	'4': FromFloat64(+0.820),

	'b': FromFloat64(+0.805),
	'd': FromFloat64(+0.755),
	'g': FromFloat64(+0.705),

	'c': FromFloat64(+0.665),
	'ç': FromFloat64(+0.635),
	'f': FromFloat64(+0.605),
	'p': FromFloat64(+0.555),

	'a': FromFloat64(+0.505),
	'à': FromFloat64(+0.405),

	'j': FromFloat64(+0.375),
	'q': FromFloat64(+0.325),

	'e': FromFloat64(+0.305),
	'è': FromFloat64(+0.205),
	'é': FromFloat64(+0.105),

	'h': FromFloat64(+0.075),
	'k': FromFloat64(+0.025),
	'i': FromFloat64(+0.005),

	'~': FromFloat64(0.000), // Símbol pels inputs finals després dels runes del mot.

	'í': FromFloat64(-0.005),

	'w': FromFloat64(-0.025),
	'y': FromFloat64(-0.075),

	'o': FromFloat64(-0.105),
	'ò': FromFloat64(-0.205),
	'ó': FromFloat64(-0.305),

	'v': FromFloat64(-0.375),
	'z': FromFloat64(-0.325),

	'u': FromFloat64(-0.405),
	'·': FromFloat64(-0.425),
	'ú': FromFloat64(-0.505),

	'x': FromFloat64(-0.555),
	't': FromFloat64(-0.605),
	's': FromFloat64(-0.665),

	'r': FromFloat64(-0.705),
	'n': FromFloat64(-0.755),
	'm': FromFloat64(-0.805),

	'5': FromFloat64(-0.920),
	'6': FromFloat64(+0.925),
	'7': FromFloat64(-0.930),
	'8': FromFloat64(-0.935),
	'9': FromFloat64(-0.940),

	'l': FromFloat64(-0.855),

	'!':  FromFloat64(-0.920),
	'?':  FromFloat64(-0.930),
	'\'': FromFloat64(-0.940),
	'}':  FromFloat64(-0.950),
	']':  FromFloat64(-0.960),
	')':  FromFloat64(+0.970),
	':':  FromFloat64(-0.980),
	'.':  FromFloat64(-0.990),

	'¬': FromFloat64(-1.000), // Rune per a símbol desconegut.
}

func FixedPointsToString(pFPs []FixedPoint) string {
	var sb strings.Builder

	for _, fp := range pFPs {
		rn := RuneFromFixedPoint(fp)
		if rn != rune(_padd) {
			sb.WriteRune(rn)
		}
	}

	return sb.String()
}

func StringToFixedPoints(pStr string, pLength int) (rFPs []FixedPoint) {
	for _, r := range pStr {
		fp, exists := Symbols[rune(r)]
		if !exists {
			fp = _errc
		}
		rFPs = append(rFPs, fp)
	}
	for len(rFPs) < pLength {
		rFPs = append(rFPs, _padd)
	}

	// fmt.Printf("MOT: '%s' LEN: %d, LEN2: %d, WANT: %d\n", pStr, len(pStr), len(rFPs), pLength)
	return
}

func FixedPointFromRune(pChar rune) (rFp FixedPoint) {
	rFp = Symbols['¬']
	for rn, fp := range Symbols {
		if rn == pChar {
			rFp = fp
			break
		}
	}
	return
}

func RuneFromFixedPoint(pFP FixedPoint) (rRune rune) {
	rRune = '¬'
	for rn, fp := range Symbols {
		if fp.Equal(pFP) {
			rRune = rn
			break
		}
	}
	return
}
