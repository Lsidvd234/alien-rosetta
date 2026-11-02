# Are word-piece starts a different kind of vector from letter starts? (`analysis/start_kind.py`)

Base model input embeddings (Qwen3-0.6B-Base), centred on the mean token. Starts = each run's saved glyph start vectors (27 glyphs).

## 1. Mean cosine between start vectors

Diagonal: two different glyphs of the same run. Off-diagonal: any glyph of one run vs any glyph of the other.

| | own letters | wrong letters (p1) | random word pieces | random vectors |
|---|---|---|---|---|
| own letters | 0.193 | 0.223 | 0.004 | 0.004 |
| wrong letters (p1) | 0.223 | 0.193 | 0.004 | 0.004 |
| random word pieces | 0.004 | 0.004 | 0.001 | -0.001 |
| random vectors | 0.004 | 0.004 | -0.001 | 0.001 |

## 2. Nearest base-vocabulary tokens of each start (cosine)

**own letters**

- glyph k: 'k' 1.00, ' k' 0.71, 'K' 0.61, '_k' 0.57, '\tk' 0.54, '(k' 0.53
- glyph aa: 'aa' 1.00, 'aaa' 0.66, ' aa' 0.62, 'AA' 0.56, 'aaaa' 0.56, '(aa' 0.46
- glyph t: 't' 1.00, ' t' 0.61, '.t' 0.50, '\tt' 0.49, 'd' 0.48, 'c' 0.48
- glyph i: 'i' 1.00, ' i' 0.59, 'a' 0.49, '_i' 0.49, 'iT' 0.49, '\ti' 0.47
- glyph m: 'm' 1.00, ' m' 0.68, '\tm' 0.57, '.m' 0.56, '_m' 0.56, 'M' 0.55
- glyph zh: 'zh' 1.00, ' zh' 0.69, 'Zh' 0.50, 'ZH' 0.47, ' Zh' 0.46, '.zh' 0.42

**wrong letters (p1)**

- glyph k: 'ii' 1.00, 'iii' 0.72, ' ii' 0.70, 'II' 0.58, '(ii' 0.51, ' iii' 0.51
- glyph aa: 'ny' 1.00, ' ny' 0.51, 'NY' 0.51, 'ni' 0.43, 'nym' 0.42, 'na' 0.40
- glyph t: 'nx' 1.00, ' nx' 0.67, 'NX' 0.54, 'Nx' 0.52, '(nx' 0.52, 'nr' 0.41
- glyph i: 'u' 1.00, ' u' 0.66, 'U' 0.58, '\tu' 0.52, '(u' 0.51, '_u' 0.50
- glyph m: 'v' 1.00, ' v' 0.73, 'V' 0.63, '_v' 0.61, '\tv' 0.58, '(v' 0.56
- glyph zh: 'n' 1.00, ' n' 0.66, 'N' 0.56, '_n' 0.54, 'm' 0.54, '.n' 0.53

**random word pieces**

- glyph k: 'Extra' 1.00, ' Extra' 0.85, 'extra' 0.77, ' extra' 0.75, '_extra' 0.71, '.extra' 0.64
- glyph aa: ' Pure' 1.00, 'Pure' 0.86, ' pure' 0.75, 'pure' 0.69, ' PURE' 0.62, '纯' 0.60
- glyph t: 'gas' 1.00, ' gas' 0.74, 'Gas' 0.72, ' Gas' 0.70, '_gas' 0.67, '气体' 0.59
- glyph i: 'ylene' 1.00, 'yl' 0.28, 'ethylene' 0.25, 'yla' 0.23, 'YLE' 0.23, 'ylan' 0.21
- glyph m: ' motifs' 1.00, ' motif' 0.77, ' themes' 0.37, ' motives' 0.33, ' patterns' 0.30, ' Mot' 0.30
- glyph zh: ' Established' 1.00, ' established' 0.71, 'Establish' 0.62, '-established' 0.57, ' Establish' 0.54, 'establish' 0.50

**random vectors**

- glyph k: '科创' 0.14, 'ocytes' 0.13, '麟' 0.12, '_interval' 0.12, ' arteries' 0.12, 'ていきたい' 0.12
- glyph aa: '_ai' 0.14, '笼' 0.14, ' gór' 0.12, '(can' 0.12, ' libc' 0.12, ' diapers' 0.12
- glyph t: 'adj' 0.15, 'orns' 0.14, 'oj' 0.14, 'Radio' 0.13, '第一个' 0.13, '烁' 0.13
- glyph i: 'locs' 0.16, '힙' 0.14, ' Moor' 0.13, '/ge' 0.13, '+");\n' 0.13, '��' 0.13
- glyph m: 'Driving' 0.14, 'Driver' 0.14, ' Driver' 0.13, '_drive' 0.13, '-pound' 0.13, ' Driving' 0.13
- glyph zh: 'toEqual' 0.13, ' الاث' 0.13, '.ARR' 0.13, ' 큰' 0.13, 'Js' 0.12, '悰' 0.12

## 3. Letter-vs-word probe

Logistic regression on base-vocabulary tokens: 3002 spelling pieces (1-2 letters) vs 12008 whole words (5+ letters), start tokens excluded. Held-out balanced accuracy 1.000.

| start set | mean P(spelling piece) | glyphs judged spelling piece (P > 0.5) |
|---|---|---|
| own letters | 0.943 | 26 / 27 |
| wrong letters (p1) | 0.943 | 26 / 27 |
| random word pieces | 0.102 | 1 / 27 |
| random vectors | 0.050 | 0 / 27 |

