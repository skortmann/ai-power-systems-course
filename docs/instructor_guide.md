# Instructor guide

Per-notebook timings, the places students reliably get stuck, what the exercises
are supposed to produce, and questions worth arguing about.

Assume **90–120 minutes of contact time per tutorial**, with the notebook run in
advance (`uv run python scripts/build_notebooks.py`) so nobody watches a progress
bar.

---

## Before the first session

- Have everyone run `uv sync` **the week before**. It downloads ~1 GB and takes
  five minutes on a good connection; it takes an hour on conference wifi.
- Decide whether you want real data. If yes, have them run
  `scripts/download_data.py` too, and note that some numbers in the committed
  notebooks will then differ from theirs. That is a feature — it is exercise 3
  of tutorial 01 — but say so in advance or you will field the same question ten
  times.
- Read [`course_overview.md`](course_overview.md). The most important thing to
  internalise before teaching: **several notebooks report results that
  contradict the folklore**, on purpose. If a student says "but I read that
  Transformers are faster than LSTMs", the notebook has a measurement and an
  explanation ready.

---

## Tutorial 01 — The Classical Machine Learning Paradigm

**Duration** 120 min (the longest; it carries the methodology for everything else).
**Prerequisites** Python, NumPy, basic statistics.

**Key concepts** features/labels, chronological splitting, leakage, bias–variance,
baselines, physical plausibility.

**Where students get stuck**

- *The lag convention.* "Lags are counted from the forecast origin, not from the
  predicted timestamp" takes a full worked example on the board. Draw a timeline.
  Everything downstream depends on it.
- *Why not a random split?* They will accept it as a rule and not believe it.
  Section 7 exists to make them believe it: the same data and model, only the
  split changes, and 1-NN's error drops 30%.
- *"Persistence and seasonal naive are the same row?"* At a 24-hour horizon on
  hourly data they are literally the same forecast. Students find this
  surprising and it is worth two minutes.

**Optional** the cross-validation section (12) can be assigned as reading.

**Discussion questions**

1. The temperature feature is a *measurement* at the forecast origin, not a
   weather forecast for the target hour. Is using it cheating? Under what
   circumstances is it defensible?
2. The PV model had $R^2 = 0.81$ and predicted non-zero generation in **every**
   night hour. Who, in an organisation, is responsible for catching that?
3. Holidays are 1.5x harder than normal days and barely move the aggregate MAE.
   What should the loss function have been?

**Expected exercise outcomes**

1. *(conceptual)* A good answer distinguishes "available in principle" from
   "available with the same accuracy", and proposes adding horizon-dependent
   noise to simulate forecast error.
2. *(coding)* The weekly blend should beat gradient boosting on holidays
   specifically while losing overall — the point is that aggregate metrics hide
   regime-specific behaviour.
3. *(research)* On real OPSD data the ranking of the learned models usually
   survives; the *gap* to the baselines narrows, because real load is noisier.
   Students who predicted otherwise should say why they were wrong.

---

## Tutorial 02 — Neural Networks and Backpropagation

**Duration** 90–120 min.
**Prerequisites** tutorial 01; partial derivatives and the chain rule.

**Key concepts** neuron, activation, backpropagation, autograd, learning rate,
early stopping.

**Where students get stuck**

- *The backward pass indices.* Work $\delta_2 = \hat y - y$ through on the board
  before they read the code.
- *Gradient checking* feels like busywork until you show them a sign error that
  still produces a decreasing loss curve. Do that live — flip a sign in
  `backward` and rerun.
- *Standardisation.* "Why does the network need it and the tree does not?" is a
  genuinely good question. Answer: gradient magnitudes scale with input
  magnitudes; a tree only compares.

**Optional** the learning-rate sweep (7.1) if time is short — but it is the most
practically useful part of the notebook.

**Discussion questions**

1. The MLP and gradient boosting see identical inputs and score within a few
   percent. Why adopt the harder one?
2. What exactly does a trained tree ensemble *store*, and why can it not be
   pretrained?

**Expected exercise outcomes**

1. A good answer says: a tree stores thresholds over *your* features and has no
   internal vector to reuse; an MLP's hidden layer is a learned representation
   that a new head can attach to.
2. The two-hidden-layer gradient check should pass to ~1e-6. Students who skip
   the check and go straight to training will report "it works" with a wrong
   gradient — let one of them do it.
3. Raw 168 values without engineered features: the MLP typically does *not*
   recover the lost information in a comparable budget. This sets up tutorial 03.

---

## Tutorial 03 — Learning from Sequences

**Duration** 120 min. **Runtime** ~4 min. **Prerequisites** 02.

**Key concepts** recurrence, hidden state, vanishing gradients, LSTM gating,
error vs. lead time.

**Where students get stuck**

- *The gradient measurement (section 6).* Three curves, and the middle one — a
  default-initialised LSTM — is **no better than the vanilla RNN**. Students
  expect the LSTM to win automatically. The lesson is that the architecture
  provides a *route* to long memory, and initialisation decides whether gradient
  descent can find it.
- *Window index arithmetic.* The notebook asserts its baselines against the raw
  series; walk through that assertion rather than skipping it.

**Discussion questions**

1. Error grows with lead time for every model. Which operational decisions
   depend on which part of that curve?
2. Section 10 reports a hypothesis that failed. Why publish it?

**Expected exercise outcomes**

1. *(conceptual)* The three-way comparison usually shows leakage dominating
   drift on this dataset, but the point is that they *measured* it rather than
   assumed.
2. *(coding)* The vanilla RNN typically stops improving around 48–72 hours of
   context; the LSTM keeps going. Watch the runtime — the recurrence is
   sequential.
3. *(research)* Recursive forecasting shows compounding error: a steeper curve
   that crosses the direct method somewhere in the horizon.

---

## Tutorial 04 — Representation Learning (the hinge)

**Duration** 120 min. **Prerequisites** 02. **Do not skip this one.**

**Key concepts** latent space, masked modelling, contrastive learning, linear
probes, label efficiency.

**Where students get stuck**

- *"Where is the label?"* There isn't one. Say it three times.
- *The linear-probe results are mixed on purpose.* PV share is readable
  ($R^2 \approx 0.7$), weekend is not ($R^2 < 0$). Students want a clean win.
  The lesson is that a representation encodes what its objective needed —
  we normalised away the level that distinguishes weekends, so it cannot be
  there.
- *The t-SNE trap.* Weekends look somewhat clustered in the projection and are
  not linearly decodable. This is the single most transferable methodological
  point in the notebook: **a 2-D projection is weak evidence.**
- *The label-efficiency gain is small* (a few percent). Do not oversell it; the
  notebook explains that 24 numbers is too easy an input for the argument to
  bite, and that this is exactly why tutorials 09 and 10 are more convincing.

**Discussion questions**

1. Section 12 shows the probe achieving $R^2 = 0.57$ on a quantity the
   preprocessing *deleted*. What is it actually exploiting, and when does that
   break?
2. If pretraining buys label efficiency rather than accuracy, when is it worth
   the compute?

---

## Tutorial 05 — Attention

**Duration** 90 min. **Runtime** ~30 s. **Prerequisites** 02; matrix products.

**Key concepts** Q/K/V, scaling, masking, positional encoding, multi-head,
the limits of attention as explanation.

**Where students get stuck**

- *Q, K and V all come from the same thing in self-attention.* Draw it.
- *"Why $\sqrt{d_k}$?"* Section 5.3 measures the softmax saturating at
  $d_k = 256$. Show the bar chart.
- *Weight ≠ contribution.* Section 5.1 now makes this concrete: a position with
  weight 0.007 and a large value supplies 37% of the output. Land this early; it
  is the foundation for section 7.2.

**Discussion questions**

1. Attention is permutation-equivariant, which is fatal for a time series and
   *desirable* for a graph. What would a positional encoding even mean for a
   bus? (Open question; see the GridFM literature.)
2. Uniform attention scored measurably worse here. Does that make the attention
   map an explanation?

---

## Tutorial 06 — Transformers

**Duration** 120 min. **Runtime** ~18 min (it trains four models). **Prerequisites** 05.

**Key concepts** the block, residuals, pre-norm, encoder/decoder, $O(n^2)$,
work versus sequential depth.

**Where students get stuck**

- *"A token is a time step?"* Yes. The table in section 4 is the whole point.
- *The timing result.* The Transformer trains **slower** than the LSTM here.
  Students who have read blog posts will object. The notebook has the arithmetic:
  a feed-forward block at every position is more work, but none of it is
  sequential, and four CPU threads cannot cash that in. This is a genuinely
  useful thing for a researcher to internalise before benchmarking anything.
- *Stride-4 training windows.* Explain the redundancy argument or someone will
  think it is a shortcut.

**Discussion questions**

1. When would you choose an LSTM over a Transformer in 2026, and on what
   evidence?
2. Zeng et al. (2022) showed a linear model beating several Transformers on
   time-series benchmarks. How should that change your priors?

---

## Tutorial 07 — Language Models

**Duration** 90–120 min. **Runtime** ~7 min. **Prerequisites** 06.

**Key concepts** tokenization, teacher forcing, perplexity, sampling controls,
scaling laws, fabrication.

**Where students get stuck**

- *Perplexity.* "How many characters is it effectively choosing between" lands
  better than the formula.
- *The fitted scaling exponent.* Four points over one order of magnitude on a
  270 kB corpus. It shows the *shape*; the number is not comparable to Kaplan's
  and the notebook says so. Do not let anyone quote it.
- *Fabrication is the objective working correctly.* Nothing in the loss rewards
  truth. This reframing does more work than any list of "LLM limitations".

**Discussion questions**

1. The model produces well-formatted, entirely invented disturbance reports.
   Which of those failures would RAG fix, and which would it not?
2. Our largest model has the lowest loss and is still not a foundation model.
   What is missing?

---

## Tutorial 08 — Pretrained LLMs and Adaptation

**Duration** 120 min. **Runtime** ~3 min plus model downloads on first run.
**Prerequisites** 07. **Needs network access** on first run.

**Key concepts** the adaptation spectrum, LoRA, in-context learning, RAG, kinds
of knowledge.

**Where students get stuck**

- *LoRA's parameter count.* The notebook asserts the formula against the actual
  count. Do that on the board too: $2 \times d \times r$ per adapted matrix.
- *"RAG trains the model."* It does not. Repeat it. Then show that the
  classifier's weights are byte-identical before and after.
- *The embedding failure (section 10)* — "95 percent loaded" and "105 percent
  loaded" get near-identical embeddings — usually produces an audible reaction.
  Sit with it. It is the most important slide in the notebook for anyone
  planning to build retrieval over engineering documents.

**Note** SmolLM2-135M defines the per-unit system as a *currency* system. That
is a real, reproducible hallucination and the RAG section fixes it. It is the
best live demo in the course.

**Discussion questions**

1. A fine-tuned 22 M-parameter encoder beat a prompted 135 M-parameter LLM here.
   When does that flip?
2. Which of the three kinds of knowledge would you accept for a protection
   setting? For a maintenance schedule? For a literature summary?

---

## Tutorial 09 — Foundation Models Beyond LLMs

**Duration** 120 min. **Runtime** ~3 min plus a ~500 MB model download on first run.
**Prerequisites** 06; 04 helps a great deal.

**Key concepts** the definition and its three requirements, the cross-domain
table, zero-shot transfer, probabilistic evaluation, when the foundation model
loses.

**Where students get stuck**

- *The five-way distinction* (pretrained / backbone / foundation model / LLM /
  generative). Spend real time here. `docs/foundation_models.md` has the table.
- *"Zero-shot" does not mean "never seen anything like it."* The synthetic
  dataset makes this checkable, which is rare — use it.
- *Wanting the foundation model to win.* It does not always, and the notebook is
  built around the cases where it does not.

**Discussion questions**

1. Chronos-2 has 120 M parameters; WindFM has 8.1 M and beats larger models
   zero-shot on wind. What does "scale" buy, then?
2. Under what distribution shift would you expect the specialist to fail first?

---

## Tutorial 10 — Toward Grid Foundation Models (capstone)

**Duration** 120 min, or a double session.
**Runtime** ~4 min. **Prerequisites** 04, 06, 09; `pandapower` helps.

**Key concepts** graphs, message passing, permutation equivariance, multi-grid
pretraining, physics-informed loss, transfer to unseen topology.

**Where students get stuck**

- *Why a GNN at all?* Because a fixed-width model trained on 30 buses cannot be
  *evaluated* on 118. Say it that way — not "graphs are natural" but "nothing
  else can even run".
- *Per-unit normalisation.* This is what makes features comparable across
  voltage levels, and without it transfer is hopeless before training starts.
- *Disappointment.* The Mini-GridFM may not beat a specialist trained on the
  target grid. That is the honest and expected outcome at this scale, and the
  notebook demonstrates the *methodology*, which is the deliverable.

**Discussion questions**

1. What should the tokens of a grid foundation model be? (No settled answer;
   `docs/foundation_models.md` §7 has the candidate list.)
2. What would have to be true before a GridFM could sit in a control room?
3. GridSFM trained on 200 grids and 500,000 scenarios. We used seven and a few
   thousand. Which of our conclusions scale and which are artefacts?

---

## Assessment suggestions

**Project (recommended).** Take one tutorial's method and apply it to a dataset
or grid the student brings. Deliverables: a notebook, a baseline comparison, a
physical-validity check, and an honest limitations section. Mark the limitations
section hardest.

**Written.** Give them a recent AI-for-energy paper and ask for a referee
report using the checklist in `docs/foundation_models.md` §6.

**Oral.** "Explain to a colleague who has not taken this course why a foundation
model is not the same thing as a large language model, using one example from
outside language."

---

## Common misconceptions to pre-empt

Listed in the README and dealt with properly in
[`docs/foundation_models.md`](foundation_models.md). The two that survive the
course most stubbornly:

- **"Bigger is better."** Counter with WindFM (8.1 M) and GridSFM (the *S* is
  for Small), plus tutorial 02's result.
- **"The attention map shows what the model used."** Counter with tutorial 05's
  ablation and the weight-versus-contribution table.

---

## If something breaks

| symptom | cause | fix |
|---|---|---|
| `uv sync` resolves slowly | no cache | it is a one-off; ~5 min |
| Hugging Face downloads fail | rate limiting | set `HF_TOKEN`, or `AI_POWER_COURSE_OFFLINE=1` to skip those sections |
| A notebook takes far longer than the table says | another notebook is also running | they are CPU-bound; run one at a time |
| Numbers differ from the committed outputs | different BLAS/thread count, or real data downloaded | expected; the *conclusions* should hold, and if they do not, that is a discussion |
| `pandapower` prints numba warnings | numba is not a dependency | already suppressed via `run_power_flow`; report it if you see one |

Rebuild everything with `uv run python scripts/build_notebooks.py`. For one
notebook or the reduced configuration, import the function instead of editing
the file:

```python
from build_notebooks import build
build(only=("07_",), fast=True)
```
