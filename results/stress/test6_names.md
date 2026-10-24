# Test 6: invented names, spelled by sound

300 invented names (never seen; not words of the language), e.g. 'Muularo dissolved'.
The name should be written in glyphs by its sound (as the first word of the translation).
Scored teacher-forced on the name's glyphs only: name fully right / share of glyphs right.

| model | name fully right | glyphs right |
|---|---|---|
| Option 1: Latin-sound start | 0.0% | 12.2% |
| Option 2: + transliteration | 0.3% | 27.2% |
| Option 3a: plain text first | 0.0% | 9.6% |
| Control: shuffled-sound start | 0.0% | 11.2% |

## Same names through the transliteration prompt ('Latin: muularo / Glyphs:')

Only option 2 was trained on this prompt (on training sentences); for the others it is unseen.

| model | name fully right | glyphs right |
|---|---|---|
| Option 1: Latin-sound start | 0.0% | 11.7% |
| Option 2: + transliteration | 78.3% | 88.4% |
| Option 3a: plain text first | 0.0% | 11.7% |
| Control: shuffled-sound start | 0.0% | 9.6% |

## By name length (glyphs)

| glyphs | names | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|---|
| 2 | 4 | 0.0% | 0.0% | 0.0% | 0.0% |
| 3 | 38 | 0.0% | 0.0% | 0.0% | 0.0% |
| 4 | 69 | 0.0% | 1.4% | 0.0% | 0.0% |
| 5 | 55 | 0.0% | 0.0% | 0.0% | 0.0% |
| 6 | 93 | 0.0% | 0.0% | 0.0% | 0.0% |
| 7 | 17 | 0.0% | 0.0% | 0.0% | 0.0% |
| 8 | 24 | 0.0% | 0.0% | 0.0% | 0.0% |

## Sample outputs (free generation, first word, glyphs read back as sounds)

| English | correct | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|---|
| Muularo dissolved | muularo | conyoyi | rxumuli | rxunengoo | nyazhen |
| Nile agreed | nile | munyanyaa | yoruy | ngaakata | vocanyii |
| Piyici smiled | piyici | noniicol | niyooy | nyaatxaangaa | nyiitay |
| Ano drifted | ano | noonongo | nxora | zhaanyevim | lximuuzha |
| Naani finished | naani | nyiiyonginx | lxaanike | nyuuvuu | tingaloo |
| Mitil loomed | mitil | lxinonxo | kinooti | txanyatoo | tiitxurxe |
| Vuumepa succeeded | vuumepa | nomokaa | vuuvoo | neruulxa | moracu |
| Rutapa punted | rutapa | nyevaazhum | ngurapi | vaaraam | ngekinee |
