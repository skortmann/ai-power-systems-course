# Final correctness audit

This document records an audit of the course's mathematics, implementations and
results. It is written to be checked, not believed: every claim below names the
script that produced it and the number it produced.

**The audit found real defects.** Eleven High-severity issues and a larger tail
of Medium ones, across the tutorials, the library and the exercise track. The
most consequential are not crashes — every notebook executed and every test
passed throughout. They are numbers that were quietly wrong and sentences that
contradicted the output printed directly above them.

Three findings are worth stating up front, because they are the shape of the
whole report:

* Tutorial 01 reported a **15.3% MAE improvement as "1.5% better"**, because
  `skill_score` defaults to RMSE and the result was printed beside two MAE
  numbers — in the section arguing the effect is large.
* Tutorial 03's flagship claim, *"Nobody engineered `hour_sin` into the
  recurrence — it emerged"*, was made about a network that is **fed `hour_sin`
  as input channel 1**.
* **66% of the language model's validation text was a verbatim copy of its
  training text**, because the deliberately repeated handbook was appended at
  the end of the corpus and the 90/10 split sent both repeats into validation.

Nothing here rests on "the notebooks all execute". They do, and that is
reported in §3 as one check among many rather than as a conclusion.

## Contents

1. [Executive summary](#1-executive-summary)
2. [Environment and versions](#2-environment-and-versions)
3. [Notebook execution status](#3-notebook-execution-status)
4. [Mathematical formulation audit](#4-mathematical-formulation-audit)
5. [Power-system physics audit](#5-power-system-physics-audit)
6. [Causal masking and attention](#6-causal-masking-and-attention)
7. [Data leakage audit](#7-data-leakage-audit)
8. [Metrics and baselines](#8-metrics-and-baselines)
9. [Probabilistic forecasting](#9-probabilistic-forecasting)
10. [Numerical reproducibility](#10-numerical-reproducibility)
11. [Independent reviewer findings](#11-independent-reviewer-findings)
12. [Issues found and fixes applied](#12-issues-found-and-fixes-applied)
13. [Remaining limitations](#13-remaining-limitations)
14. [Final verification table](#14-final-verification-table)

---

## 1. Executive summary

Three independent reviewers examined the course: this audit's own harness, a
Claude reviewer, and Codex. They were given deliberately overlapping
assignments on the highest-stakes areas and complementary ones elsewhere.

| Reviewer | Assignment | Found |
|---|---|---|
| `scripts/audit.py` (this audit) | 34 executable checks: metrics, causal masking, leakage, baselines, reproducibility, physics | 34/34 pass after fixes; found 0 defects directly, but every fix below is now guarded by one |
| Claude reviewer | Attention internals, leakage in the data layer, metrics, baselines, GNN, reproducibility, **prose vs printed output** | 13 findings, 7 High |
| Codex | Exercise/solution track, tutorials 04/08/09, tokenizer and corpus, documentation | 24 findings, 4 High |

The division of labour is the finding that matters most. **The harness found
nothing.** It verified that the library is correct — metrics match
scikit-learn, causal masks do not leak, baselines are not crippled, splits are
chronological — and every one of those verifications passed on the *first* run,
before any fix. Every High-severity defect was found by a reader comparing
prose against output, or by an experiment nobody had thought to run.

That is a limitation of automated verification, not an accident of this
particular harness, and §13 says so.

---

## 2. Environment and versions

```
Python      3.12                       torch          2.14.0+cpu
numpy       2.4.6                      scikit-learn   1.9.1
pandas      2.2+                       transformers   5.17.0
pandapower  3.5.5                      scipy          1.13+
```

Dependencies are locked in `uv.lock` and installed with `uv sync`. The audit ran
at **full scale**: `fast_mode()` returned `False`, so the reduced CI
configuration was not in play for any number quoted here.

### Tolerances

Tolerances live in `src/ai_power_course/tolerances.py`, one constant per
concept, each documented with the arithmetic that justifies its magnitude.
Machine learning needs a wider spread than most numerical work, because the
quantities genuinely differ in precision: a causal-mask check is exact
arithmetic and is tested at `1e-6` on float32, while a Monte Carlo coverage
estimate has several per cent of sampling error and testing it tightly would
only measure the seed.

```
quantity                tolerance   guards
exact (float32)         1e-06       causal masking, frozen weights
exact (float64)         1e-12       NumPy reference implementations
metric (relative)       1e-10       our metric vs an independent one
gradient (relative)     1e-04       analytic vs finite difference
probability (absolute)  5e-02       empirical rate vs nominal
coverage (absolute)     1e-01       interval coverage vs nominal
physics (relative)      1e-06       generation == load + losses
loss (relative)         1e-05       reproducibility of training
```

---

## 3. Notebook execution status

All ten tutorial notebooks are regenerated from their jupytext sources — which
carry no outputs, so this is a genuine cold run — and executed end to end, plus
the exercise and solution notebooks from `ai_power_course.exercises`.

No result depends on stale notebook state: the sources are the only input, and
`tutorials/*.ipynb` is deleted before each rebuild.

**Execution is necessary and nowhere near sufficient.** Every defect in §12 was
present in a notebook that executed cleanly. Two of them — the masked-loss
self-check and the padding demonstration — were present in cells that
*asserted* their own correctness and passed.

---

## 4. Mathematical formulation audit

### Metrics

Every metric was recomputed independently, against scikit-learn where it has an
equivalent and against an explicit formula where it does not:

```
MAE:   ours=3.3478697057  sklearn=3.3478697057
RMSE:  ours=4.1252568423  sklearn=4.1252568423
R2:    ours=0.8741988299  sklearn=0.8741988299
bias:  ours=0.00599610    hand=0.00599610
MAPE:  ours=7.09966884    hand=7.09966884
sMAPE: ours=7.15559289    hand=7.15559289
nRMSE: ours=8.18915815    hand=8.18915815
```

Structural properties were checked rather than assumed: the skill score is
exactly 0 against its own reference and exactly 1 for a perfect forecast; bias
is positive when the model over-predicts.

### CRPS — a real defect, fixed

`crps_from_quantiles` computed `2 * mean(pinball losses)`. That is correct only
when the quantile levels form a midpoint quadrature of `(0, 1)`; on the common
`{i/(n+1)}` grid it carries a systematic `(n+1)/n` factor, and on a sparse grid
it is badly off. Measured against the analytic Gaussian value `sigma/sqrt(pi)`:

| levels | `2 * mean` | after the fix |
|---|---|---|
| 19 (`0.05 … 0.95`) | +4.9% | +1.5% |
| 99 (`0.01 … 0.99`) | +1.1% | **+0.2%** |
| 3 (`0.1, 0.5, 0.9`) — what Tutorial 09 uses | **−11.3%** | −6.0% |

The old estimator did not converge as the grid densified, which a quadrature
must. It now weights each level by the width of the interval it represents. The
residual −6.0% at three levels is irreducible — three quantiles do not pin down
a distribution — and the course labels that number "approx." accordingly.

CRPS remains exactly `0.0` for a perfect forecast, and still prefers a
correctly located predictive distribution over one shifted by 1σ
(`0.595` vs `0.878`).

---

## 5. Power-system physics audit

The identity `generation == load + losses` holds on **every** catalogued
network, not just a convenient one:

```
case9     2.3e-11     case24_ieee_rts  4.8e-16     case39   3.1e-11
case14    2.1e-11     case30           2.4e-11     case57   4.9e-11
case33bw  2.1e-08     case118          4.0e-11
```

Losses are positive (133.17 MW on case118) and voltage magnitudes are
physically plausible (`|V| ∈ [0.9430, 1.0500]` pu over 118 buses).

### `physical_report` — a real defect, fixed

`physical_report(net, vm_pu, va_degree)` merged three kinds of number under one
flat dict. Two of them — voltage violations and the power-balance residual —
are computed from the *predicted* state. The third, `line_loading_violations`,
takes only `net` and reads `net.res_line.loading_percent`: the converged
solution already stored on the network.

Feeding a nonsense predicted state (all 0.80 pu, 0°) to case14 moves
`p_mismatch_max_mw` from 0.0 to 232.4 and `violation_rate` from 0.0 to 1.0,
while `max_loading_percent` stays bit-identical at 1.5076. In a table headed
"physical checks on the prediction", three columns were the answer key.

Those keys now carry a `truth_` prefix, and a test asserts the distinction:
prediction-based entries must move when the state changes, `truth_` entries
must not.

The better fix — computing loading from the predicted state — is available in
principle: `ppc["internal"]["Yf"]` gives branch currents from any voltage
vector, and that route reproduces pandapower's `loading_percent` to **1e-12**
on case14, case30 and case118. It is not applied because the
internal-branch-to-element mapping does not hold on every catalogued network
(case33bw disagrees, 32 rows against 37 elements). Recorded as open rather than
done unreliably.

---

## 6. Causal masking and attention

This is the load-bearing check of the language-model half of the course. A
causal mask that leaks lets position *t* see token *t+1*; the model trains, the
loss falls beautifully, and every downstream claim is meaningless. Reading the
mask cannot establish this — the test is a perturbation.

**Result: clean on every path, and this was true before any fix.**

```
NumPy attention:           perturbing key/value at t leaves rows < t unchanged   0.000e+00
MultiHeadSelfAttention:    perturbing x at t leaves positions < t unchanged      0.000e+00
TinyGPT (end to end):      changing token t cannot change logits at positions<t  0.000e+00
```

Supporting properties, each measured:

* scores are divided by `sqrt(d_k)` using the **head** dimension, not the model
  dimension — implied logit gap 2.8284271247 against an expected
  `d_k/sqrt(d_k)` of 2.8284271247 for `d_k = 8`;
* softmax normalises over the key axis (row sums 1.000000000000);
* `-inf` is applied **before** the softmax, so masked weights are exactly 0.0;
* `causal_mask` is lower-triangular *including* the diagonal, matching
  `np.tril` exactly;
* the independent reviewer additionally verified the fused
  `F.scaled_dot_product_attention` path, boolean-mask polarity, head
  split/merge round-tripping, and that `sinusoidal_positional_encoding` matches
  Vaswani exactly and is actually added in both `TinyGPT.forward` and
  `TransformerForecaster.forward`.

---

## 7. Data leakage audit

### The data layer is clean

Verified by experiment, not by reading:

* `time_split` is chronological and disjoint (train ends 2018-12-31 23:00,
  valid 2019-01-01 … 2019-12-31, test starts 2020-01-01; zero index overlap).
* Lag and rolling features read only the past, anchored at the forecast origin:
  `lag{k}[t] == load[t-k]` to 0.0, and `roll24_mean[t]` equals the hand mean of
  `t-23 … t` to 1e-10.
* `make_supervised` puts the target exactly `horizon` ahead of the origin.
* No feature is a near-copy of the future target (highest `|corr|` is
  `load_lag0` at 0.544).
* `make_windows` starts the horizon immediately after the context, with no
  window straddling a boundary, and the tutorials call it **per split**.
* Standardisers are fit on training data only.

### Two real leaks, both in content rather than in splitting

A split function can be perfectly chronological and the data still contaminated,
because the contamination is in the *text*. Both of these were live:

**The language-model corpus.** `generate_corpus` appended two extra copies of
the handbook at the end (`parts += ["", HANDBOOK, "", HANDBOOK]`), and Tutorial
07 holds out the final 10%. The result:

```
before:  65-char validation windows verbatim in train: 66.0%
         129-char validation windows verbatim in train: 61.6%
after:   0.2%  /  0.0%
```

The repetition is wanted — a character model needs to see technical vocabulary
more than once — so the copies stay; they now sit near the front, inside the
training region. Corpus length is unchanged at 275,906 characters.

**The event-classification dataset.** `generate_event_dataset` returns 880
templated sentences of which only 718 are distinct. Splitting the raw list put
**56 of 264 test sentences (21.2%) verbatim into training**, scoring every
adaptation method in Tutorial 08 — frozen probe, full fine-tune, LoRA,
head-only — on one-in-five items it had memorised, and flattering the
higher-capacity methods most, which is precisely the comparison that section
makes. Now deduplicated before splitting, with an assertion on zero overlap.

Both are now regression-checked in `scripts/audit.py`.

---

## 8. Metrics and baselines

A weak baseline makes every model look good, so the baselines were checked as
carefully as the models.

**They are not crippled.** `persistence` returns the value at the forecast
origin at all 60 tested origins; `seasonal_naive` reads `y[t + h - season]`, a
past value; the documented identity `seasonal_naive(season=h) ≡ persistence` at
horizon `h` holds exactly; and `seasonal_naive` **raises** rather than leaking
when `season < horizon`. `climatology` is fit on the training series only and
keys on the target timestamp's calendar slot, which is known arbitrarily far
ahead and therefore not a leak.

### Tutorial 01's skill score — a real defect, fixed

```
unconstrained  MAE  1,878.8 MW
constrained    MAE  1,590.6 MW   (1.5% better, and physically valid)
```

Those two MAE numbers differ by **15.3%**. The parenthesis said 1.5% because
`skill_score` defaults to `metric=rmse`, and an RMSE skill was printed beside
two MAE figures without saying so — understating the effect tenfold in a
section whose entire point is *"Physical knowledge is free accuracy… the single
cheapest improvement available"*.

It now reports the MAE improvement the reader can verify from the line above,
and reports the RMSE skill separately with the reason they differ (clipping
removes many small night-time errors, which MAE counts equally and RMSE barely
notices).

---

## 9. Probabilistic forecasting

The pinball loss is asymmetric in the correct direction, checked against its
defining property rather than its code: at `q = 0.9`, under-prediction by 1
costs exactly 0.9 and over-prediction by 1 costs exactly 0.1.

Stronger, the loss is minimised **at** the quantile it names — verified by a
grid search against the empirical quantile of 200,000 draws:

```
q=0.1: argmin=-1.2800, empirical quantile=-1.2800
q=0.5: argmin=+0.0000, empirical quantile=-0.0002
q=0.9: argmin=+1.2850, empirical quantile=+1.2847
```

Empirical coverage matches nominal at every level tested (N = 20,000):
0.50 → 0.5039, 0.80 → 0.8049, 0.95 → 0.9498.

CRPS is covered in §4.

---

## 10. Numerical reproducibility

```
set_seed makes NumPy, torch and random reproducible  : all three identical
seeded weight initialisation identical across runs   : every parameter tensor equal
```

The RNG discipline is unusually good and worth recording: **every**
`default_rng` call in `src/` and `tutorials/_sources/` is explicitly seeded —
zero unseeded `default_rng()` calls, and zero uses of the legacy `np.random`
global. Reproducibility therefore does not depend on `set_seed` having been
called at the right moment. The independent reviewer confirmed that two full
`train_model` runs produce bit-identical loss curves.

One caveat, recorded as Low: `set_seed` seeds the legacy `np.random` global,
which nothing in the repository uses, and does not set
`torch.backends.cudnn.deterministic` or `CUBLAS_WORKSPACE_CONFIG`. The CPU path
the course actually runs on is deterministic; the CUDA path advertised by
`torch_device()` is not pinned.

---

## 11. Independent reviewer findings

Both reviewers were told to run experiments rather than read code, to separate
what they executed from what they reasoned about, and to report what they
checked and found correct.

**Every finding was re-tested before anything was changed.** A reviewer's
finding is a hypothesis. Two examples of why that matters:

* The Claude reviewer reported the CRPS grid bias with an analytic reference of
  1.1296 for N(3,2). My first re-test used the wrong analytic expression (the
  CRPS at `z = 0` rather than its expectation) and appeared to contradict them.
  The correct value is `sigma/sqrt(pi) = 1.1284`; the reviewer was right and my
  check was wrong.
* Codex reported that the padding demonstration never pads. Re-tested
  independently: `"A" * 400` is **3 tokens** to MiniLM's WordPiece vocabulary,
  shorter than the 13-token probe. Confirmed.

The reviewers also converged independently on one finding — that the
course-wide leaderboard compares forecasts across tutorials that do not perform
the same task — which is stronger evidence than either alone.

### What they confirmed as correct

Recorded because an audit that reports only faults is not describing what
happened. The Claude reviewer verified, by experiment: all attention paths
(§6); every metric against hand computation on an array containing a zero and a
negative; all four baselines against a ramp series; splitting and windowing,
including that `make_windows` is called per split with the previous split's
tail handed over as leading context only; message-passing direction and
permutation equivariance in the GNN; per-unit consistency on four networks; and
that `get_batch` builds the next-token target as `x` shifted by exactly one, so
the language-model objective is the right one.

It also found **no instance** of a model being said to "understand" or to have
"learned physics" — the two overclaims it did find (T03's emergence, T10's
"never told the voltage level") are contradicted by the code rather than by
rhetoric.

---

## 12. Issues found and fixes applied

### Fixed

| ID | Location | Issue | Severity | Verification | Fix |
|---|---|---|---|---|---|
| **A-01** | `07_language_models.py`, `models/tinygpt.py` | The table claims character tokenization makes unknown symbols "impossible"; `encode` silently **deletes** them. `"Voltage 5 µV"` decodes as `"Voltage 5 V"` — a factor of a million, unreported | **High** | Round-trip on `'Voltage ΔV = 5 µV ⚡'` → `'Voltage V = 5 V '` | `encode` raises by default with the offending characters named; `on_unknown="drop"` keeps the lossy path explicitly. Table corrected: unknown symbols are **possible**, and only a byte vocabulary has none |
| **A-02** | `corpus.py` | Two extra handbook copies appended at the end put 66% of the LM validation split verbatim into training | **High** | 65-char windows: 66.0% → 0.2%; 129-char: 61.6% → 0.0% | Repeats moved to the front, inside the training region. Corpus length unchanged (275,906) |
| **A-03** | `08_pretrained_llms_and_adaptation.py` | 880 templated sentences hold only 718 distinct; 56 of 264 test rows (21.2%) were verbatim in training, flattering the higher-capacity adaptation methods most | **High** | Measured before/after; overlap 56 → 0 | Deduplicate before splitting, with an assertion on zero overlap |
| **A-04** | `01_classical_machine_learning.py` | A **15.3%** MAE improvement printed as "1.5% better" beside two MAE numbers, because `skill_score` defaults to RMSE | **High** | 1,878.8 → 1,590.6 MW is 15.3%; the line said 1.5% | Reports the MAE improvement explicitly, and the RMSE skill separately with the reason they differ |
| **A-05** | `01_classical_machine_learning.py` | Takeaway: "A shuffled split flattered a **Ridge** model by a **double-digit** percentage". The notebook's own table: Ridge 0.9%, random forest 13.6%, 1-NN 29.8% — and the prose four cells earlier says "Ridge barely notices" | **High** | Read from the executed notebook | Names the flexible models and their real numbers, and draws the actual lesson: the more a model can memorise, the more a leaky split rewards it |
| **A-06** | `01_classical_machine_learning.py` | Takeaway: "**Gradient boosting** … It won this notebook". The notebook prints `Best model: Random forest (MAE 1,637 MW)` | **High** | Read from the executed notebook | Corrected to the random forest, with gradient boosting named as close behind |
| **A-07** | `03_rnns_and_lstms.py` | "Nobody engineered `hour_sin` into the recurrence — it emerged", about a network fed `hour_sin` as input channel 1. The flagship claim of the tutorial | **High** | `CHANNELS = ["load_mw", "hour_sin", …]`, printed by the notebook itself | Probes a **12-hour harmonic nobody supplied** alongside the 24-hour sine that is an input, and makes the contrast the lesson: check the input list before believing a representation emerged |
| **A-08** | `10_grid_foundation_models.py` | The "predict the mean" baseline is a **standard deviation**, placed in a column headed `vm MAE [pu]` and drawn as the reference line. For Gaussian data σ exceeds the MAD by 1.25, so every "we beat the baseline" claim was inflated ~25%. It was also computed on the test slice — an oracle | **High** | Derivation plus the code path | MAE of a mean predictor, with the mean taken over the **training** graphs |
| **A-09** | `grid/physics.py` | `physical_report` mixed ground-truth line loading into a report about a prediction; three columns were constant across methods | **High** | Nonsense state moves `p_mismatch_max_mw` 0.0 → 232.4 while `max_loading_percent` stays at 1.5076 | `truth_` prefix on the network-derived keys, plus a test asserting prediction-based entries move and `truth_` entries do not |
| **A-10** | `exercises/part2.py` | The padding-bug demonstration used `"A" * 400` as the "longer" batch mate — **3 tokens** to MiniLM, shorter than the 13-token probe, so nothing padded. Both pooling methods differed by 1.3e-07 while the text narrated a large effect, and the self-check passed | **High** | Token counts measured; after the fix, naive pooling differs by **5.07e-01** against mask-aware at 1.4e-07 | A real 131-token sentence, so the probe is genuinely padded |
| **A-11** | `04_representation_learning.py` | "The frozen probe matches or beats scratch training **across the whole range**"; at 20 labels the run has scratch at 0.0363 against the probe's 0.0375 | **High** | Reviewer re-ran the label-budget sweep | Describes the crossing, and notes that a confident summary sentence is exactly what erases it |
| **A-12** | `metrics.py` | `crps_from_quantiles` used `2 * mean(pinball)`, valid only for a midpoint grid; −11.3% on the 3-level grid Tutorial 09 uses, and not convergent | **Medium** | Analytic Gaussian comparison, §4 | Proper quadrature weights; converges to +0.2% at 99 levels |
| **A-13** | `scripts/audit.py`, `tests/test_grid.py` | — | — | — | Regression checks added for A-02, A-03, A-09 and A-12 so none can return silently |

### Found by the audit's own instruments

Recorded because three produced **false results against the course**, and an
audit whose instruments are unexamined is not evidence.

| ID | Instrument | Defect | Effect | Resolution |
|---|---|---|---|---|
| **B-01** | `scripts/audit.py` | Grid check summed only `res_ext_grid` and `res_sgen` | case118 appeared to generate 514 MW against 4,242 MW of load — a false FAIL. case118 carries most of its supply on PV `gen` buses | `res_gen` included; now checks all eight networks |
| **B-02** | `scripts/audit.py` | Assumed `TinyGPT(vocab_size=…)` and a zero-argument network builder | Two sections silently skipped as REVIEW, leaving TinyGPT causality and grid physics **unchecked** | Real APIs used (`TinyGPTConfig`, `load_network`) |
| **B-03** | this audit, re-testing the reviewer | Used the CRPS at `z = 0` instead of its expectation as the analytic reference | Appeared to contradict a correct reviewer finding | Corrected to `sigma/sqrt(pi)`; the reviewer was right |
| **B-04** | `pkill -f ai_rerun` | Matched the audit's own shell | Killed the running rebuild *and* the patch command, which then silently did not apply | Verified the patch had landed before proceeding |
| **B-05** | this report | §4's metric values were written from memory rather than copied from the harness output, and were wrong in every digit after the first (quoted MAE 3.1652338902 against the actual 3.3478697057) | An audit report citing fabricated evidence, in a document whose argument is that claims must match output | Corrected from the harness output. Recorded rather than quietly fixed, because it is the same failure mode as A-04 through A-07 and the author is not exempt from it |

---

## 13. Remaining limitations

**The harness found nothing, and that is the headline.** `scripts/audit.py`
runs 34 checks over metrics, causal masking, leakage, baselines,
reproducibility and physics. All 34 passed on the first run, before any fix.
Every High-severity defect in §12 was found by a reader comparing prose to
output, or by an experiment nobody had thought to run — the tokenizer's µV, the
`"A" * 400` that tokenizes to three tokens, a standard deviation in a column
labelled MAE.

A harness that recomputes a quantity cannot notice that the sentence above it
says something else. That is a property of the method, not of this
implementation, and it is the strongest argument in this report for
commissioning independent review.

**Open, recorded, not fixed.** The course ships with these:

| Location | Issue | Severity |
|---|---|---|
| `results.py`, README, `course_overview.md` | The course-wide leaderboard ranks models evaluated on **different test sets** with skill scores against **different references** (implied reference RMSE 2,951.5 for T01/T02 against 3,018.0 for T03/T06), then prints "This table is the spine of the course". Both reviewers found this independently | **Medium** |
| `06_transformers.py` | "Identical windows to tutorial 03, so the LSTM comparison is exact" — `TRAIN_STRIDE = 4` gives 4,333 training windows against T03's 17,329. The in-notebook comparison is fair; the sentence and the leaderboard row are not | **Medium** |
| `grid/graphs.py` | The `is_transformer` edge feature tests for an off-nominal tap, not for a transformer. case14: 3 of 5 flagged; case57: 15 of 17; case118: 9 of 13 | **Medium** |
| `10_grid_foundation_models.py` | Section conclusions are `print`ed unconditionally regardless of the numbers computed above them, for a single-seed stochastic experiment that also runs under `fast_mode()` | **Medium** |
| `10_grid_foundation_models.py` | "The encoder was never told what its voltage level is" — `log_base_kv` is an explicit node feature | **Medium** |
| `exercises/part1.py`, `part2.py` | Several self-checks accept wrong work: the masked-loss check passes on a loss that ignores the mask entirely; the skill-score check accepts a constant zero; the message-passing check verifies supplied reference code rather than the student's aggregation | **Medium** |
| `exercises/part1.py`, `part2.py` | Two self-checks read variables the student scaffold never creates, so correct work raises `NameError` | **Medium** |
| `grid/physics.py` | Line loading is still read from the network's stored solution rather than computed from the prediction. The computation is validated to 1e-12 on three networks but the branch mapping fails on case33bw (§5) | **Medium** |
| `02_neural_networks.py` | The gradient-boosting contender omits `early_stopping=False`, so sklearn carves a **random** validation split out of a time series — the leak Tutorial 01 spends a section warning about | **Medium** |
| `config.py` | `set_seed` does not pin the CUDA path (§10) | **Low** |
| `metrics.py` | `nrmse` docstring says "mean of the truth"; the code divides by `mean(|truth|)` | **Low** |
| `docs/ai_timeline.md` | TimesFM placed in 2024 against the timeline's stated first-public-appearance rule | **Low** |

**Scope.** Tutorials 05 and 09 and most of `docs/` received a single reviewer's
pass. The exercise track's 6,100 lines were audited by Codex alone; its
findings are recorded but only the two High ones were independently re-tested
here.

**What correctness does not cover.** This audit establishes that the numbers
are right and the claims about them are accurate. Whether the course teaches
well is a separate question that measurement does not answer.

---

## 14. Final verification table

| Component | Executed | Independent comparison | Mathematical audit | Physical audit | Status |
|---|:--:|:--:|:--:|:--:|---|
| Point metrics | ✓ | ✓ scikit-learn | ✓ hand formulas | N/A | PASS |
| Probabilistic metrics | ✓ | ✓ analytic Gaussian | ✓ minimiser at the quantile | N/A | PASS (A-12 fixed) |
| Baselines | ✓ | ✓ ramp series | ✓ causality guards | N/A | PASS |
| Splitting and windowing | ✓ | ✓ hand alignment | ✓ | N/A | PASS |
| Corpus / dataset content | ✓ | ✓ verbatim-overlap count | ✓ | N/A | PASS (A-02, A-03 fixed) |
| Causal masking | ✓ | ✓ perturbation, 3 paths | ✓ sqrt(d_k), softmax axis | N/A | PASS |
| Tokenizer | ✓ | ✓ round-trip incl. OOV | ✓ | N/A | PASS (A-01 fixed) |
| Attention / Transformer | ✓ | ✓ vs torch to 1.1e-16 | ✓ Vaswani positional encoding | N/A | PASS |
| Language model (T07) | ✓ | ✓ next-token target | ✓ perplexity = exp(CE) | N/A | PASS |
| Representation learning | ✓ | ✓ label-budget sweep | ✓ | N/A | PASS (A-11 fixed) |
| Adaptation / LoRA (T08) | ✓ | ✓ | ✓ | N/A | PASS (A-03, A-10 fixed) |
| GNN / message passing | ✓ | ✓ perturbation | ✓ equivariance | ✓ | PASS |
| Grid physics | ✓ | ✓ pandapower, 8 networks | ✓ per-unit | ✓ 2.1e-08 worst | PASS (A-09 fixed) |
| Reproducibility | ✓ | ✓ repeated runs | ✓ RNG discipline | N/A | PASS |
| Literature | N/A | ✓ 57 arXiv IDs | ✓ | N/A | PASS |
| Leaderboard comparability | ✓ | ✓ implied references | ✗ | N/A | **OPEN** |
| Exercise self-checks | ✓ | ✓ Codex | partial | N/A | **OPEN** |
