r"""Hand-written text for the results report. Every number here is taken from the result files that the
report also prints (results/final_20k.md, results/curve_summary.md, results/stress/test*.md)."""

PREFACE = r"""
\section*{How to read this report}
Every table in this report is converted automatically from the result files written by the experiment and test
scripts (the source file is named above each table); none were retyped. The text in blue \emph{Inference} boxes is
my interpretation of the table directly above it, and orange \emph{Caveat} boxes flag known limits of a test. All
stress tests use the four saved 50k-sentence models (hard language, invented glyphs, one new token per glyph):
\textbf{Option 1} (each glyph starts at the embedding of its Latin sound), \textbf{Option 2} (copy start +
transliteration examples), \textbf{Option 3a} (copy start + a plain-glyph-text phase first) and the
\textbf{Control} (each glyph starts at a deliberately wrong, shuffled sound). Unless stated otherwise, numbers in
the text refer to Option 1.
"""

SETUP = r"""
\begin{itemize}
\item \textbf{Model:} Qwen3-0.6B-Base, fine-tuned with LoRA: rank 16, alpha 32, dropout 0, on all linear layers
(q, k, v, o, gate, up, down); 3 epochs; learning rate $2\times10^{-4}$; effective batch 32 (16 $\times$ 2 or
32 $\times$ 1); fp16 on a Kaggle T4. One seed per run, except the 20k grid (seeds 0, 1, 2).
\item \textbf{Prompt:} \texttt{English: <sentence>} newline \texttt{Target:} newline, then the translation. Scored by
greedy decoding; \emph{exact} = the whole sentence is right; \emph{content-word accuracy} = every token of each
content word is right (fair across scripts).
\item \textbf{Languages:} invented by a generator (\texttt{conlang/}). \emph{Easy} = English-like (SVO, articles, few
endings). \emph{Hard} = Tamil-like: SOV, no articles, case endings (object, location), every tense an ending, full
person/number/gender agreement, a negation ending, irregular past stems, and sound changes where endings join
(sandhi). A new seed gives a new random language with the same grammar.
\item \textbf{Scripts:} Latin letters; real Tamil letters; and \emph{invented glyphs} from the Unicode Private Use
Area, which no model has seen (3 byte-tokens each in Qwen's tokenizer, or one new token each when added).
\item \textbf{Data:} a 50,000-sentence training pool (runs use the first 5k/10k/20k/50k) and 500 sentences per test
split, regenerated exactly from the seed.
\end{itemize}
"""

SPLITS_INTRO = r"""
Each test sentence is new (never in training). The splits differ in what is held out of training entirely:
\begin{itemize}
\item \texttt{test\_iid}: ordinary new sentences, same kind as training.
\item \texttt{test\_comp}: a verb + object pair never seen together in training (10\% of verb--object pairs are
held out, chosen by a hash of the pair).
\item \texttt{test\_morph}: past tense + negation + plural subject together; this ending combination never occurs in
training (each part does).
\item \texttt{test\_long}: an adjective on both nouns plus a location; this sentence shape never occurs in training.
\end{itemize}
The table below is generated from the data generator itself (hard language, seed 0): the first two items of each
split, the English input, the gloss of the correct output, and the output written in Latin letters (the glyph
version spells exactly the same sounds, one glyph per sound or syllable).
"""

SPLITS_AFTER = r"""
Note the SOV order and the endings in the glosses: \texttt{-ACC} marks the object, \texttt{-LOC} the location,
\texttt{-PL} plural; the verb carries tense (\texttt{PST}/\texttt{PRS}/\texttt{FUT}), agreement (\texttt{AGR.3sgn}
etc.) and negation (\texttt{NEG}).
"""

MAIN_INTRO = r"""
The research question: \emph{does the number of tokens a script costs predict how well a model learns a new
language written in it?} And for the hardest case (hard grammar + invented glyphs), what makes it learnable?
"""

CURVES = r"""
\begin{itemize}
\item Nothing is learned at 5k sentences (all cells $\le$ 0.001 content-word accuracy); at 50k every cell reaches
0.980--0.997. Learning is a sharp jump between 10k and 50k.
\item \textbf{Token count does not predict accuracy.} Easy-Tamil costs 3.4$\times$ the tokens of easy-Latin (42.0 vs
12.5 tokens per sentence at 20k) but is as accurate (0.830 vs 0.817 content-word accuracy).
\item \textbf{Grammar matters more than script.} At 20k the hard language is far behind the easy one in both scripts
(Latin 0.048 vs 0.817; Tamil 0.232 vs 0.830).
\item In the hard language, Tamil script learns faster than Latin at 20k (0.232 vs 0.048) although it costs 2.2$\times$
the tokens (43.4 vs 19.3); at 50k both are about equal (0.990 vs 0.980). So this is a difference in learning speed,
not in what is finally learned.
\end{itemize}
"""

FINAL = r"""
\begin{itemize}
\item \textbf{Seeds:} across three random languages per cell (20k), the spread is small (sd 0.010--0.023) and the
ordering is the same in every seed: hard-Tamil (mean 0.241) is about 6$\times$ hard-Latin (0.038).
\item \textbf{Cause tests (hard, 20k):} splitting Latin into one character per token \emph{doubles} accuracy
(0.048 to 0.106) despite 2.2$\times$ the tokens; at the same token count, Tamil letters double it again (0.232 at 43.4
tokens vs 0.106 at 42.9). Accented Latin, one character per token, is worst (0.016). So both splitting and the
script itself matter, and more tokens can help.
\item \textbf{Invented glyphs as 3 byte-tokens:} nothing at 20k in either language. The easy language recovers at 50k
(0.992); the hard language is slow: 0.446 at 35k, 0.636 at 50k, 0.898 at 80k, and even at 80k it fails new
structures (\texttt{test\_long} 0.048) and new ending combinations (\texttt{test\_morph} 0.464).
\item \textbf{Invented glyphs as one new token each (copy start):} hard 50k reaches 0.755 content-word accuracy but
0.000 on \texttt{test\_long}: fewer tokens alone do not fix generalisation.
\item \textbf{The fixes (hard, 50k, one token per glyph):} Option 1 (Latin-sound start) gives 0.990 / 0.986 / 0.972 /
0.974 on iid / comp / morph / long, better than the Latin-script and Tamil-script targets, in 47.8 minutes.
Option 2 (transliteration) also works (0.894 on long) at 2.2$\times$ the training time (103.5 min). Option 3a fails
(0.002 on comp and long). The Control (wrong, shuffled sounds) matches Option 1 in distribution (0.982 iid) but not
on new structures (morph 0.860, long 0.650).
\item \textbf{Main conclusion:} the difficulty of invented glyphs is not the number of tokens but that the new tokens
start with \emph{no meaning}. A distinct, letter-like start gives most of the in-distribution gain; starting at the
\emph{correct} sounds is what gives generalisation to new structures.
\end{itemize}
"""

FIGURES = [
    ("fig1_learning_curves.png", "Figure 1. Learning curves (content-word accuracy vs training sentences)."),
    ("fig6_generalisation_fixes.png", "Figure 6. The fixes for hard grammar + invented glyphs, by test split."),
    ("fig7_word_frequency.png", "Figure 7. Accuracy by how often a word was seen in training (stress test 2)."),
]

STRESS_INTRO = r"""
Thirty behavioural tests probe where the saved models break (in the spirit of CheckList, Ribeiro et al.\ 2020).
Each builds new sentences with the generator, so the correct answer is always known exactly. Scoring is either
teacher-forced (each next token given the correct prefix), free generation (what the model actually writes, greedy),
or contrastive (which of two translations the model gives higher probability). Test 12 is invalid and is kept only
for completeness (see its caveat).
"""

GROUPS = [
    ("What the model learned (tests 1--3, 9, 13--15, 20)", [1, 2, 3, 9, 13, 14, 15, 20]),
    ("English-side variation (tests 4, 8, 19, 22, 23, 30)", [4, 8, 19, 22, 23, 30]),
    ("Structures never seen in training (tests 5, 6, 7, 10, 25)", [5, 6, 7, 10, 25]),
    ("Output, decoding and confidence (tests 11, 12, 16, 17, 18, 24, 29)", [11, 12, 16, 17, 18, 24, 29]),
    ("Model health and side effects (tests 21, 26)", [21, 26]),
    ("Predicting failures from the input (tests 27, 28)", [27, 28]),
]

CAVEATS = {
    5: r"""Training never contained two adjectives on one noun, so the ``correct'' translation for levels 4--6 is built
from an \emph{assumed} rule (each adjective is its own word before the noun, in English order). Test 25 checks this:
the models do not prefer this or any other rule-following version.""",
    6: r"""The names are capitalised, while all training English is lower case; that alone is a change the model
never saw.""",
    7: r"""The generator never produced a transitive verb without an object, so the 0\% row is a gap in the training
data, not a failure of role assignment.""",
    12: r"""\textbf{This test is invalid.} Its ``confidence'' is the probability of the \emph{correct} translation,
which is unknown in real use, and its notion of ``right'' comes from the same probabilities, so a separation of 1.00
is guaranteed. Test 17 replaces it, using the model's confidence in its \emph{own} output.""",
    20: r"""A bug in the first run of this test (both adjective+noun groups used seen pairs) was fixed, and the test
was rerun; the table is from the fixed run. The ``noun in a role it never had'' check is empty because every noun
appears as both subject and object in training.""",
    21: r"""English perplexity is measured on only 30 everyday sentences, and perplexity is extremely sensitive to
probability placed on glyph tokens. The test was run with each model in its own process after an earlier run crashed
(out of Windows memory); runs are deterministic, so this does not change results.""",
    27: r"""The flags are computed on words split at whitespace, so changes inside the whitespace (double spaces,
no-break spaces) are invisible to them. Word classes come from the generator's own lexicon; function words are kept as
themselves.""",
    28: r"""One Gaussian (with a small shrinkage term) fitted on the mean hidden state of 2,000 training prompts; other
detectors (relative Mahalanobis, per-token features) might do better.""",
    29: r"""The pooled numbers depend on the mix of inputs: most of this pool is deliberately off-template (accuracy
38.9\% for Option 1). The in-distribution rows (normal + the 4 test splits) show the usual-use case.""",
    25: r"""The candidate translations are the plausible ways to handle a second adjective; the model may prefer
something outside this list (shown as ``none of these'' in what it wrote).""",
    26: r"""Same 30 English sentences as test 21. ``Start'' rows put the glyph embeddings back to the values they had
before training (the Latin sounds for Option 1, random-word copies for Options 2 and 3a, wrong sounds for the
Control).""",
}

TEST = {
    1: r"""Option 1's glyphs stay at their sounds after training (nearest sound = own sound for 100\%, structure match
0.99) while moving 27\% of their length. Options 2 and 3a and the Control never acquire a sound layout (their nearest
ordinary tokens are unrelated words such as `Extra', `imageSize'). So Option 1 keeps the free dictionary it was given;
the others learn glyph identities without any sound structure.""",
    2: r"""The baseline (random-word start) needs about 26--35 sightings before a word is learned (1.1\% at 1--15,
84.0\% at 26--35). Option 1 is at 99.8\% even for words seen 1--15 times; the Control is at 89.7\% there. A meaningful
start makes rare words learnable.""",
    3: r"""Endings combine systematically: 99.1\% on combinations seen in training and 95.0\% on the held-out
combination (past + negated + plural subject), against 71.7\% for Option 2 and 84.7\% for the Control. Option 3a is
at 0.1--0.2\% whenever there is an object but 97.5\% without one: its failure is the transitive verb, not the
endings.""",
    4: r"""Surface changes that keep the training template are harmless (capitalised + full stop 99.7\%), but anything
the model never saw on the English side breaks it: passive 0.0\%, UPPER CASE 17.7\%, synonyms 18.7\%, ``a'' for
``the'' 48.3\%. All four models fail the same way, so this is a property of the narrow training English, not of the
glyph start.""",
    5: r"""The held-out long shape is handled (95.3\%, and 96.0\% with the held-out ending combination at the same time;
Control 66.7\% and 50.0\%). Two or more adjectives on one noun fail completely (0.0--0.7\%) for every model.""",
    6: r"""No model can write an unseen name inside a sentence (0.0--0.3\% fully right). Option 2, which was trained on
a transliteration prompt, spells 78.3\% of names correctly through that prompt: the skill exists but is not used
inside translation.""",
    7: r"""Roles are assigned by structure, not by memory: swapping subject and object of a training sentence gives
99.3\%, and the same noun as subject and object 97.7\%. A transitive verb without an object gives 0.0\% for every
model (a shape absent from training).""",
    8: r"""Prompt and spacing details matter a lot: double spaces 4.0\%, a typo 7.2\%. Option 1 is far more robust to
prompt changes than the others: `Input/Output' labels 87.6\% (Option 2 28.0\%, Control 34.0\%), a `Translate:'
line 91.6\% (Option 2 32.0\%, Control 68.4\%).""",
    9: r"""Hard word forms are handled: irregular past 96.0\% (regular 99.3\%), and sound changes do not hurt (99.0\%
with 2+ sound changes).""",
    10: r"""Two sentences in one input are never both translated (0.0\% for every model). An unknown English word as the
object does not derail the rest of the sentence (99.3\% right otherwise, word count always right).""",
    11: r"""What the model writes equals what teacher-forced scoring predicts (99.0\% both), with no outputs too long or
too short; the cheaper teacher-forced scoring used in several tests is therefore valid. The reverse direction (glyphs
to English, never trained) is 0.0\%.""",
    12: r"""Not interpreted (invalid test; see the caveat).""",
    13: r"""No agreement attraction: the verb agrees with the subject 98.7--100\% of the time and with the intervening
object 0.0\%, even with a location or an adjective in between; in scoring, the correct verb is preferred 100\% of the
time. (Option 3a's 0\% is its broken transitive verbs, not attraction.)""",
    14: r"""Changing one English feature changes exactly the expected part of the output: 99.2--100\% right, 0.0\%
ignored, 0.0\% collateral changes, for all nine features (tense, negation, location, subject number, subject noun,
adjective, object number, pronoun person/number, pronoun gender).""",
    15: r"""Given the correct translation and a version with exactly one error, every model prefers the correct one 100\%
of the time for all eight error types, usually by large margins (e.g. 119.6 log-probability for missing sound
changes). The weakest is a regularised irregular verb (margin 6.9; Control 87.5\%, Option 3a 37.5\%), but that row
has only 8 items.""",
    16: r"""The model almost never leaves the glyph script (at most 5.0\% of outputs contain any non-glyph character), but
it also never refuses: greetings, French, code, a question, an empty input and a prompt injection all produce 1--5
confident glyph words.""",
    17: r"""Failures are \emph{model} errors, not decoding errors: beam search (5) gives the same accuracy as greedy, and
almost no wrong output is a search error (0.0--3.7\%; only the `Input/Output' prompt reaches 16.7\%). Confidence in
the model's own output separates right from wrong well when pooled (0.93), but wrong outputs still average 0.47--0.69
confidence. End-of-sentence is placed correctly (probability 1.00 on normal input).""",
    18: r"""The order rule is never broken (``verb not last'' 0.0\% everywhere). Failures fall into three types: wrong
stems (passive 80.7\%, synonym 90.7\%, UPPER CASE 92.0\%, typo 90.7\%, transitive verb without object 100\%), dropped
words (2 adjectives 74.7\%, two sentences 100\%) and extra words (double spaces 92.0\%).""",
    19: r"""Rewordings that keep the template are fine (`did X' 98.0\%, `does X' 99.3\%, full stop and capital 98.7\%).
Anything else fails: filler in front 0.0\%, object fronted 0.0\%, location fronted 0.0\%, numerals 0.0\%,
`please translate:' 4.7\%, this/that/these/those 6.0\%, Title Case 44.7\%, `is going to' 50.0\%. All models behave
alike.""",
    20: r"""Every one of the 18 sentence templates is 96.2--100\%, including the one never seen in training (97.5\%;
Control 65.0\%). Adjective+noun pairs never seen together are 100\% right in all 5 sentences each. Changing only the
subject noun changes another word in 1.0\% of cases. Accuracy holds for long outputs (97.7\% at 51+ tokens; Control
80.8\%).""",
    21: r"""Translation fine-tuning damaged plain English: perplexity on everyday sentences rises from 20.3 (base) to 372
(Option 1), 934 (Control), about 8.9 million (Option 2) and 13.2 million (Option 3a). Switching the adapter off
restores 20.5--23.2. Glyphs leak into continuations of plain English (46.7\% for Option 1, 100\% for the others).
The tokenizer is sound (0 round-trip errors in 52,000 targets), no test sentence copies a training sentence, glyph
embeddings have normal size (median 1.07--1.12 vs 0.97 for Qwen's vocabulary) and even the rarest glyph was seen
24,572 times. The Control's glyph mix-ups do not follow its wrong starting sounds (2.6\% vs 3.7\% by chance).""",
    22: r"""The model translates word by word from the English tokens: for a typo or a synonym, 94.8\% / 97.2\% of wrong
sentences have \emph{only} the changed word wrong, and 98.0\% / 99.2\% of the other words are right. UPPER CASE
changes every word's tokens, so most sentences get two or more wrong words.""",
    23: r"""Passive is read as ``first noun = doer'' by every model: all four score the subject--object-swapped
translation above the correct one in 100\% of sentences, and their outputs share 52--86\% of words with the swapped
translation but only 4--11\% with the correct one. This is the classic position heuristic (McCoy et al.\ 2019).""",
    24: r"""Within one input type, own-output confidence still separates right from wrong for normal inputs (0.96), UPPER
CASE (0.92), typos (0.92) and double spaces (0.98): it is not just detecting odd inputs. It fails for synonyms
(0.63; Option 2 0.42; Control 0.11, i.e.\ more confident when wrong) and for two adjectives (0.68). A threshold that
keeps 95\% of correct normal outputs flags 97.8--100\% of the wrong outputs for every input type except synonyms
(67.4\%) and normal inputs (50.0\%, of very few wrong outputs).""",
    25: r"""The failure is real and not caused by the assumed gold rule: the best-scoring candidate for Option 1 is ``both
adjectives, noun dropped'' (74.5\%; the second adjective takes the noun's place), for Options 2 and 3a ``first
adjective only'' (98.5\%, 79.5\%). Every model forces two adjectives into the one-adjective slot of its template.""",
    26: r"""The English damage sits in the LoRA adapter, not in the glyph rows: with the adapter off, perplexity is
20.5--23.2 for every model; with it on, removing glyphs from the prediction still leaves 247 (Option 1), 693 (Control),
88,597 (Option 3a) and 136,791 (Option 2). Probability stolen by glyphs is the smaller part (Option 1: 372 to 247).
Translation needs both parts: the adapter with the starting glyph rows translates 0\% for every model.""",
    27: r"""A single ``unseen local structure'' rule explains much but not all. The word-class-triple flag catches
81.7\% of Option 1's failures (accuracy 23.2\% when flagged vs 68.0\% when not), and every token-level change
(passive, UPPER CASE, typos, fronting, numerals, fillers, determiners, look-alike letters) is flagged on 90--100\% of
inputs. It misses failures with nothing new in them: double spaces (1.0\% accuracy, 0\% flagged) and a transitive
verb without an object (1.0\%, 0\% flagged), where something is \emph{missing} rather than new. And some unseen
patterns are handled well: `did X' 98.3\%, `does X' 100\%, full stop and capital 100\%. Unseen \emph{word pairs} do not
matter at all (64\% of normal sentences contain one, all translated correctly): the model composes at the word level.""",
    28: r"""The input alone already shows most off-template English: middle-layer distance from the training inputs
separates passive (0.93), fronting (0.96), numerals (0.97), UPPER CASE and double spaces (1.00), invisible Unicode
(0.95--1.00) and odd inputs (0.98) from normal inputs. It cannot see synonyms (0.54) or a verb missing its object
(0.46), the same blind spots as output confidence. To predict a wrong output, output confidence is better (0.95) than
input distance (0.80), and combining them does not help (0.93). Input distance is still useful where there is no
correct answer at all: flagging odd inputs (0.98) gives the model a way to refuse.""",
    29: r"""On normal use the model is well calibrated and accurate (in distribution: 99.0\% accuracy, calibration error
0.029). On the mixed pool it is over-confident: outputs with 0.8--0.9 confidence are right only 50.0\% of the time and
the calibration error is 0.166; Platt scaling on a held-out half reduces it to 0.057. Ranking by confidence does not
give a usable selective translator on off-template inputs: only 3.0\% of the pool can be answered at 99\% accuracy
(10.2\% at 95\%), because some wrong outputs (e.g.\ synonym substitutions) are among the most confident.""",
    30: r"""Invisible characters break translation like typos: look-alike Cyrillic letters 27.0\%, zero-width space
34.0\%, soft hyphen 34.0\%, full-width letters 16.0\%, three mixed changes 1.0\%. Standard NFKC normalisation fixes
full-width letters and no-break spaces (restored 100\%), removing invisible format characters also fixes zero-width
spaces and soft hyphens (restored 100\%, accuracy back to 100\%), but neither fixes look-alike letters (restored
0\%; that needs a confusables map). A one-line input clean-up is a cheap, real fix for most of this test.""",
}

OVERALL = r"""
\subsection*{1. The research question}
\textbf{Token count does not predict accuracy; the meaning a token starts with does.} Tamil script costs 2.2--3.4$\times$
the tokens of Latin and learns as well or faster; splitting Latin into single characters (more tokens) helps the hard
language; invented glyphs fail as 3 byte-tokens \emph{and} as single new tokens with an uninformative start. Starting
each glyph at its Latin sound (Option 1) gives 0.990 / 0.986 / 0.972 / 0.974 on the four splits, above both real
scripts, and the shuffled-sound Control shows that the \emph{correct} sounds are what produce generalisation to new
structures (long 0.974 vs 0.650).

\subsection*{2. What Option 1 really learned}
Within the kind of English it was trained on, Option 1 learned the invented grammar as rules: held-out ending
combinations 95.0\% (test 3), the never-seen sentence shape 97.5\% (test 20), never-seen adjective+noun pairs 100\%
(test 20), one-feature changes 99.2--100\% with no side effects (test 14), no agreement attraction (test 13), the
correct form preferred over every one-error version 100\% (test 15), and rare words learned from 1--15 sightings
(99.8\%, test 2). Its glyph embeddings keep their sound layout (test 1), its outputs are well formed (tests 11, 16),
and it is well calibrated in distribution (test 29).

\subsection*{3. Where it fails, and one mechanism behind most failures}
It fails whenever the English leaves the training template: passive 0\%, fronting 0\%, fillers 0\%, numerals 0\%,
determiners 6\%, synonyms 12--19\%, typos 3--10\%, UPPER CASE 4--18\%, invisible Unicode 1--34\% (tests 4, 8, 17, 19, 22,
24, 27, 30; the range covers every test that measured it);
and on structures absent from training: two adjectives on a noun, two sentences, a verb without its object, names
(tests 5, 6, 7, 10). Three follow-up tests show how: it translates \emph{word by word from the English tokens} (an error
stays on the changed word in 95--97\% of cases, test 22), and it fills a \emph{fixed positional template}: a passive
sentence is read as ``first noun = doer'' (swapped translation scored higher in 100\% of cases, test 23), and two
adjectives are forced into a one-adjective slot (test 25). An ``unseen local structure'' flag catches 81.7\% of the
failures (test 27); what it misses are inputs where something is missing rather than new (a verb without its object)
or changed inside the whitespace. \textbf{All four models fail in the same ways}, so these are consequences of the
narrow, single-template training English, not of the glyph start.

\subsection*{4. Can the failures be caught or fixed without retraining?}
\begin{itemize}
\item \textbf{Not by decoding:} beam search does not help and almost all errors are model errors (test 17).
\item \textbf{Partly by confidence:} the model's confidence in its own output separates right from wrong (0.93--0.95,
also within input types, tests 17, 24, 28), but it is over-confident off-template and blind to synonym errors
(0.63; tests 24, 29), so at 99\% accuracy only 3.0\% of a mixed off-template pool could be answered (test 29).
\item \textbf{Refusal:} the model never refuses (test 16), but input distance flags odd inputs with 0.98 AUROC
(test 28), a usable switch.
\item \textbf{Input clean-up:} NFKC + removing invisible characters fully repairs zero-width spaces, soft hyphens,
no-break spaces and full-width letters (test 30). Look-alike letters need a confusables map. (Lower-casing as a fix
for UPPER CASE is plausible but was not tested.)
\end{itemize}

\subsection*{5. Side effect: forgetting English}
Fine-tuning damaged plain English (perplexity 20.3 to 372 for Option 1, to millions for Options 2 and 3a; test 21).
The damage is in the LoRA adapter, not the glyph rows (test 26): adapter off restores 20.5--23.2, and removing glyphs
from the prediction removes only the smaller part of the damage. The adapters of the copy-start models damaged English
100--500$\times$ more; why is not established (one hypothesis: their adapters had to do more of the work of
identifying glyphs).

\subsection*{6. Retraining with targeted data (R1): fixes what it targets, and close variants}
R1 (Option 1 + varied English rewrites + new structures + refusals + plain-English replay, each with a held-out
variant) keeps the original splits (0.986 / 0.988 / 0.970 / 0.982) and generalises \emph{within} the patterns it was
given: passive with verbs never seen in passive 0.94, held-out fillers / determiners / numerals / spacing 0.98--1.00,
refusal of unseen kinds of non-English 0.85. It does not reach neighbouring cases that were not given: object fronting
0.00 (location fronting 0.99), unseen synonyms 0.15, three adjectives 0.30, three sentences 0.06; typos stay at 0.13.

\subsection*{7. Representative English (E1) and an analysis step}
E1 replaces the single template with English generated from grammar families (voice, word order, tense expression,
negation form, determiners, adjectives, location preposition, clause joining, orthography, lexical choice), sampled
uniformly, with 20\% of variants held out at random before training. Both versions were trained for 3 and for 5 epochs;
E1b was also repeated on a second random language (seed 1).
\textbf{The analysis step was useful and made the model converge faster:} after 3 epochs the version that first writes
an analysis of the English (E1b) already scored 0.96 / 0.93 / 0.93 / 0.84 on the original splits, while the plain
version (E1a) on the same data and schedule was at 0.26 / 0.22 / 0.27 / 0.06; E1a needed 5 epochs to reach the same
level (0.98 / 0.97 / 0.96 / 0.85). At 5 epochs the gap on most tests closes (E1b 0.99 / 0.97 / 0.97 / 0.87, seed 1
0.97 / 0.97 / 0.96 / 0.91), but E1b keeps an edge on both seeds where generalisation is hardest: five or six adjectives
(never trained) 0.71 / 0.75 vs 0.53, dropped-letter typos 0.54 / 0.57 vs 0.42. Held-out surface variants reach
0.76--0.86 for both. \textbf{Structure types that never appear in training still fail for every model, at every
length of training and with or without the analysis} (strict groups, only items that contain the structure: passive
0.00--0.02, clefts 0.01--0.09, object topicalisation 0.00--0.01, names as subject 0.00--0.02), and so do unseen synonym
pairs (0.10--0.17).

\subsection*{8. Forgetting is fixed by replay}
With about 10\% plain-English passages mixed into training, English perplexity is back near the base model
(everyday sentences 22.7--23.5 vs 20.3; WikiText-2 34.3--35.6 vs 22.9) instead of 372 / 4,208 for Option 1, and glyphs no
longer leak into English (0\% vs 47\%). Scaling the adapter down at inference is not an alternative: E1b translates
almost nothing below full strength (0.02 on the original iid split at strength 0.75).

\subsection*{9. Summary}
\begin{enumerate}
\item What makes an unseen script learnable is the meaning its new tokens start with, not their number.
\item A fine-tuned model generalises within what its data covers: new words, new combinations, more of the same,
new surface forms. Structure types absent from training are not reached, whether the data is targeted (R1) or broad
(E1). The training distribution should therefore cover every structure type the language should handle, chosen in
advance from a grammar, not from test failures.
\item An explicit analysis step was useful: it made the model converge faster (3 epochs instead of 5 for the same
scores) and kept a consistent edge on the hardest generalisation (counting beyond the trained range, typos) on two
random languages. It does not help with structure types absent from training.
\item Plain-English replay removes forgetting; refusal of non-English generalises to unseen kinds (0.64--0.85).
\end{enumerate}
"""
LIMITS = r"""
\begin{itemize}
\item \textbf{One seed per 50k model.} Only the 20k grid has three seeds (sd 0.010--0.023). Differences of a few
points between the 50k models may be noise; the large gaps (e.g.\ long 0.974 vs 0.650) are not.
\item \textbf{One invented language} (seed 0) for all stress tests; another random language with the same grammar
might differ in detail.
\item \textbf{Small cells:} stress-test rows have 60--300 items (test 15's irregular-verb row has 8), so a difference
of a few points within a row is not meaningful.
\item \textbf{Assumed gold} for structures never in training (two adjectives); test 25 shows the conclusion does not
depend on it.
\item \textbf{Forgetting} is measured on 30 everyday sentences by perplexity, which is very sensitive to probability on
glyph tokens.
\item \textbf{Synthetic English:} all English comes from the generator's templates; real English would be more varied
and would likely expose more failures.
\item \textbf{Untested hypotheses} are marked as such in the text (why copy-start adapters forget more; lower-casing as
a fix).
\item Test 12 is invalid and is not used anywhere in the conclusions.
\item \textbf{R1 and E1a are single runs (seed 0); E1b (5 epochs) has two seeds.} E1 used 30k translation inputs
(Option 1 and R1: 50k); E1's held-out list is one random draw. The E1a/E1b edge at 5 epochs (about 20 points on two
groups of 100) compares two E1b seeds against one E1a seed.
\item \textbf{The 5-epoch runs ran on a RunPod V100} (torch 2.6) instead of a Kaggle T4 (torch 2.11); all other
settings are identical. An E1b epoch costs more compute than an E1a epoch (longer outputs: 74 vs 54 training minutes for
3 epochs on the T4), so the faster convergence is about 1.2$\times$ in compute rather than 5/3 in epochs.
\item \textbf{Per-epoch cells use subsets} (100 items per original split, 30 per other group: about $\pm$9 points).
\item \textbf{A sealed set of independently written English} has not been scored yet; all English so far is generated.
\end{itemize}
"""

RETRAIN_INTRO = r"""
After the stress tests, two retraining strategies were tried, both from the base model (Qwen3-0.6B-Base, fresh LoRA,
glyphs starting at their Latin sounds, seed 0, 3 epochs; E1 was later repeated with 5 epochs and E1b with a second
seed), both with refusal examples (non-English input gives the
output \texttt{<not-english>}) and plain-English replay passages (FineWeb-Edu) against forgetting:
\begin{itemize}
\item \textbf{R1, targeted:} the template data plus English rewrites and new structures aimed at the failures found,
each with a \emph{held-out} variant never used in training (e.g.\ passive with half of the verbs; numerals one--three
trained, four--five held out).
\item \textbf{E1, representative:} English generated from grammar families sampled uniformly; 20\% of variants held
out at random before training. \textbf{E1a} translates directly; \textbf{E1b} first writes an analysis of the English
(\texttt{S=big+dog.PL V=see.PAST.NEG O=cat L=house => <glyphs>}). Same data otherwise.
\end{itemize}
"""

LOCAL_FIXES_INF = r"""Both fixes repair a failure completely when it is the only thing wrong with the input. Splitting lifts Option 1
from 0.00 to 1.00 on two sentences and to 0.98 on three (never seen), and R1 from 0.16 to 1.00 on three; lower-casing
lifts the casing variants of R1's categories to 0.96--1.00 for Option 1 and R1. They do not help when the input also
differs in other ways: the E1 groups (5 clauses, sentence case, the in-distribution mix) are E1-style English with
several kinds of variation at once, and there Option 1 and R1 stay at 0.00--0.34 with or without the fix. E1a gains
little anywhere (these are the 3-epoch E1 models; E1a was undertrained). E1b already handles multiple sentences without splitting (0.88 / 0.82); splitting
helps it on five clauses (0.30 to 0.54), which matches the error breakdown above (long outputs were its main
translation error), and lower-casing lifts its unseen casing variants from 0.72 to 0.98. Cheap input normalisation is
worth applying in front of any of these models, but it cannot replace representative training data."""

RETRAIN_PARTS = [
    ("R1 compared with Option 1", "r1_vs_option1.md",
     r"""Teacher-forced scores on 100 new items per category. ``[train]'' / ``[heldout]'' refer to R1's training data.
Option 1 never saw any of these variants, so its column shows the starting point.""",
     r"""R1 keeps the original splits within $\pm$1 point and fixes what it targets, generalising to held-out variants of
the trained patterns (passive 0.94, numerals and determiners 0.98--1.00, refusal 0.85). It does not generalise to
cases outside them: object fronting 0.00, unseen synonyms 0.15, three adjectives 0.30, three sentences 0.06, typos
0.12--0.13."""),
    ("Option 1, R1, E1a and E1b on the same test items", "e1_comparison.md",
     r"""Free generation, glyph part only. Rows starting `R1' are R1's categories with R1's train/held-out labels; for E1,
three adjectives and three sentences are inside its training range. E1's own held-out variants are listed above the
table. E1 used 30k translation inputs and was not converged (see the per-epoch table below).""",
     r"""After 3 epochs on identical data, the analysis step (E1b) is far ahead of direct translation (E1a) almost
everywhere (original iid 0.96 vs 0.26; three sentences 0.80 vs 0.02). Most of this gap is speed of learning, not a final
difference: with 5 epochs E1a catches up on most tests (section ``E1 with 5 epochs'' below). E1b also counts beyond its training range
(five or six adjectives 0.58) and handles held-out surface variants (sentence case 0.74, the determiner `one' 0.74,
dropped-letter typos 0.51). R1 is best on what it targeted (passive 0.94, synonyms it was trained on 0.74). No model
handles a structure type it never saw: counting only items that really contain it, E1's held-out passive is 0.00,
clefts 0.00--0.05 and object topicalisation 0.02--0.04 (the pooled rows in the table are inflated by items whose variant
fell back to normal word order; correction in results/e1\_heldout\_corrected.md); names as subject 0.00."""),
    ("Correction: held-out structures, counting only items that contain them", "e1_heldout_corrected.md", "",
     r"""Unseen structure types are essentially never handled by either E1 model (0.00--0.05 on the items that really
contain them). Items whose variant fell back to the ordinary form are translated much better (up to 0.89), which is
what inflated the pooled group scores. Future runs build these groups only from items that really contain the
variant."""),
    ("Forgetting, adapter scaling and per-epoch checkpoints", "followups_e1.md",
     r"""WikiText-2: 40 chunks of 256 tokens (never trained on). Scaling and per-epoch cells use subsets (50 and 30 items
per group; 100 per original split), about $\pm$7--9 points.""",
     r"""Replay removes forgetting (everyday-English perplexity 22.7--23.5 vs 20.3 base and 372 for Option 1; no glyph
leakage). Scaling the adapter down is not a free fix: translation appears only at full strength. R1 has learned most
of what it learns by epoch 2; E1a and E1b are still improving steeply at epoch 3 (E1b original iid 0.06 / 0.39 / 0.94),
so the E1 runs need more epochs (done: next section). Refusal is learned first (0.83--0.93 after epoch 1)."""),
    ("E1 with 5 epochs, and a second E1b seed", "e1_5ep.md",
     r"""5-epoch runs on a RunPod V100 (fp16, torch 2.6, transformers 5.16.1), otherwise the same settings as the 3-epoch
Kaggle runs. Held-out structure rows are strict for the 5-epoch runs and left blank for the 3-epoch runs (their pooled
values are not comparable). Seed 1 is a different random language, so its test items differ. 100 items per E1 group:
about $\pm$8--10 points.""",
     r"""\textbf{The analysis step was useful and made the model converge faster.} With the analysis, 3 epochs were enough
(E1b original splits 0.96 / 0.93 / 0.93 / 0.84); without it, the same data and schedule left E1a at
0.26 / 0.22 / 0.27 / 0.06, and E1a needed 5 epochs to reach that level (0.98 / 0.97 / 0.96 / 0.85). Counting compute
rather than epochs, the saving is smaller but still real: an E1b epoch costs about 1.4$\times$ an E1a epoch (longer
outputs), so 3 E1b epochs (74 T4-minutes) against 5 E1a epochs (about 90 T4-minutes) is about 1.2$\times$ less compute. At 5
epochs the two versions are close on most tests, and E1b's two seeds agree within about 5 points on most rows, so the
results hold on a second random language. E1b keeps an edge on both seeds on the hardest generalisation: five or six
adjectives 0.71 / 0.75 vs 0.53, typos 0.54 / 0.57 vs 0.42. Longer training does not help unseen structure types:
passive, clefts, object topicalisation and names as subject stay at 0.00--0.09 for every model. With representative
data and enough training, the model generalises to new surface forms and to more of the same structure, but not to a
structure type it has never seen."""),
    ("Where E1b goes wrong: understanding vs translation", "e1b_error_breakdown.md",
     r"""For each item: is the analysis E1b writes exactly right, and is the glyph output right? Held-out structure rows
here use the pooled (uncorrected) groups.""",
     r"""E1b's failures on unseen structures are understanding errors: the analysis is already wrong in 79--97\% of
held-out passive, cleft, topicalised, name-subject and synonym items, and the glyphs then follow the wrong analysis.
When the analysis is right, the glyphs are almost always right (translation errors 0--7\% on most groups). The
exception is long output: on five clauses 32\% of items have a correct analysis but wrong glyphs, and on the original
long split 14\%. Pooled over all groups: 70\% both right, 7\% translation errors, 23\% understanding errors. Better
English understanding, not better glyph writing, is what the remaining errors need."""),
    ("Inference-time fixes without retraining: sentence splitting and lower-casing", "local_fixes.md",
     r"""50 items per group, free generation. Splitting breaks the input at `.', `;', `!' and `, and', translates each
part and joins the outputs. Lower-casing changes the input only.""",
     LOCAL_FIXES_INF),
]

PROVENANCE = r"""
\begin{itemize}
\item Main tables: \texttt{results/curve\_summary.md} (\texttt{analysis/curve\_table.py}) and
\texttt{results/final\_20k.md} (\texttt{analysis/final\_20k.py}), built from the per-run JSON files in
\texttt{results/kaggle/<run>/results/}. Figures: \texttt{analysis/plots.py}.
\item Stress tests: \texttt{analysis/stress\_tests.py} (1--6), \texttt{stress\_tests2.py} (7--12),
\texttt{stress\_tests3.py} (13--21), \texttt{stress\_tests4.py} (22--26), \texttt{stress\_tests5.py} (27--30); each
writes \texttt{results/stress/test<N>\_*.md}. Saved models: \texttt{results/kaggle/s1-hard-latininit-50k} (Option 1),
\texttt{s2-hard-translit-50k} (Option 2), \texttt{s3-hard-plaintext-50k} (Option 3a),
\texttt{s4-hard-shuffledinit-50k} (Control); the reruns reproduced the original runs' scores exactly.
\item Split examples in Section 1 are produced by \texttt{conlang.dataset.build} at report time.
\item Retraining: data from \texttt{train/augment.py} (R1) and \texttt{train/represent.py} (E1); runs
\texttt{results/kaggle/r1-hard-aug-50k}, \texttt{e1a-hard-rep-50k}, \texttt{e1b-hard-rep-cot-50k} (each with
\texttt{model/}, \texttt{model\_epoch1/}, \texttt{model\_epoch2/}). Comparisons: \texttt{analysis/compare\_r1.py},
\texttt{analysis/compare\_e1.py}, \texttt{analysis/followups\_e1.py}, \texttt{analysis/compare\_5ep.py}
(5-epoch runs \texttt{e1a-hard-rep-5ep}, \texttt{e1b-hard-rep-cot-5ep}, \texttt{e1b-hard-rep-cot-5ep-seed1}, trained
on RunPod with \texttt{kaggle\_tools/runpod\_e1.sh}), \texttt{analysis/local\_fixes.py}. Literature behind the E1 design:
\texttt{results/generalisation\_research.md}.
\item This report: \texttt{docs/results\_report/build\_report.py} (tables) and \texttt{inferences.py} (text).
\end{itemize}
"""
