# Instructor guide

Per-notebook timings, the places students reliably get stuck, what the exercises
are supposed to produce, and questions worth arguing about. The separate
**exercise track** — `tutorials/exercise.ipynb` and its solutions — has its own
section below, with per-chapter timings and the mistakes students actually make.

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

## The exercise track

[`tutorials/exercise.ipynb`](../tutorials/exercise.ipynb) has one chapter per
tutorial, 44 tasks in total.
[`tutorials/solution.ipynb`](../tutorials/solution.ipynb) works every one of them
through and explains why the implementation looks the way it does. Both are
generated from `src/ai_power_course/exercises/`, so if you edit a task, rebuild
with `uv run python scripts/build_exercise_notebooks.py` — that command also
executes the solution notebook end to end, so a broken answer fails the build
instead of reaching a student.

**Setting them.** One exercise chapter per tutorial, as homework, is the
intended cadence. Budget roughly the tutorial's own runtime again for the
compute, plus the times below for the thinking. The reflection questions are the
ones to open the next session with; they are written to disagree about, and
several have more than one defensible answer.

**Marking.** The self-check cells verify shapes and invariants, not reasoning.
For the analysis tasks, mark the interpretation, not the number — a student who
reports that their hypothesis failed has done the exercise correctly. For the
reflection tasks there is no key; the solution notebook gives one strong answer
and says so.

| Chapter | Time | Hardest | The one to spend the session on |
|---|---|---|---|
| 01 Classical ML | 60 min | ★★ | 1.5 — why a good model loses to persistence |
| 02 Neural networks | 60 min | ★★ | 2.4 — the MLP does *not* clearly win |
| 03 RNNs and LSTMs | 75 min | ★★★ | 3.4 — a hypothesis that fails |
| 04 Representation learning | 75 min | ★★★ | 4.3 — the weekend probe fails, and why |
| 05 Attention | 90 min | ★★★ | 5.1 — the central implementation of the course |
| 06 Transformers | 90 min | ★★★ | 6.2 — equivariance, and chapter 10 wanting the opposite |
| 07 Language models | 75 min | ★★★ | 7.5 — "just autocomplete" |
| 08 Pretrained LLMs | 60 min | ★★★ | 8.4 — scenario 4, fine-tune versus retrieve |
| 09 Foundation models | 75 min | ★★★ | 9.4(c) — what the experiment does *not* support |
| 10 Grid foundation models | 120 min | ★★★ | 10.6 — the open question |

Times are contact-equivalent for a student who has done the tutorial, excluding
compute. Chapters 08 and 09 download pretrained models; have students run the
setup cells before the session or set `AI_POWER_COURSE_OFFLINE=1`, which makes
those tasks report and skip while the rest still runs.

---

### Chapter 01 — Classical Machine Learning · 5 tasks · ★–★★

**Objective** Frame a forecasting problem, split it so the split cannot lie, and
build a baseline worth beating.

**Likely mistakes**

- *Counting lags from the target instead of the forecast origin* in 1.2. This is
  the leak, and it is invisible: the table builds, the model trains, the score is
  impossibly good. The task's assertion against the raw series catches it.
- *Fitting the scaler before splitting* in 1.4. The `Pipeline` makes it
  structurally impossible; some students will still try to scale up front.
- *Forgetting the hourly-to-TWh factor* in 1.1. Ask what changes at 15-minute
  resolution.

**Discussion** 1.5 asks for three mechanisms by which a good model loses to
persistence. Students reliably find distribution shift and miss "the evaluation
is not measuring what you think". Push on that one.

---

### Chapter 02 — Neural Networks · 4 tasks · ★–★★

**Objective** Write a gradient step by hand, then earn `loss.backward()`.

**Likely mistakes**

- *Forgetting `in_features = width` inside the layer loop* in 2.2.
- *Calling `zero_grad` after `backward`* in 2.3. It trains, badly, and nothing
  errors. Ask them to predict what the loss curve would look like.
- *Reporting MAE in standardised units* in 2.4. The check catches it with a
  magnitude assertion; make sure they understand why 0.3 "MW" was the giveaway.

**Discussion** 2.4 is deliberately anticlimactic — the MLP and gradient boosting
land within a few percent. The point is the argument in the printed output: a
tree ensemble has no hidden layer to reuse, nothing to transfer, and no way to
train without labels. That is the thread to chapter 04.

---

### Chapter 03 — RNNs and LSTMs · 4 tasks · ★–★★★

**Objective** Move from a feature table to the sequence itself.

**Likely mistakes**

- *Off-by-one in `create_sequences`* (3.1). The self-check compares against
  `np.arange`, which makes the error obvious; without it students lose an hour.
- *Dropping the channel axis.* `nn.LSTM` then reads the sequence length as a
  feature count and the error message is unhelpful.
- *Confusing `h_n[-1]` with `output[:, -1, :]`.* Identical for one layer, not for
  a stack.

**Discussion** 3.4 is the first task where the intuitive hypothesis is wrong:
recent volatility does *not* predict error on this data. Students dislike
reporting a negative result about their own idea. Insist on it — the quartile
comparison alongside the correlation is the habit being taught.

**Note** the accuracy assertion in 3.4's self-check is skipped under
`AI_POWER_COURSE_FAST=1`, because two epochs cannot beat persistence.

---

### Chapter 04 — Representation Learning · 4 tasks · ★★–★★★

**Objective** The hinge of the course: train with no labels and no task, then
find the representation useful for a question it never saw.

**Likely mistakes**

- *Scoring the reconstruction everywhere instead of only at masked positions*
  (4.1). The loss falls beautifully and the representation learns nothing.
- *Using a constant zero as the mask token.* Zero is a plausible value for a
  standardised profile, so it is indistinguishable from real data.
- *Pretraining on everything, then probing.* A leak, and a subtle one — the
  encoder has seen the evaluation profiles, just without their labels.

**Discussion** 4.3's weekend probe **fails**, and the reason is three cells
earlier: every day was normalised to zero mean, and a weekend differs mostly in
*level*. This is the single most valuable moment in the chapter. Preprocessing
and pretraining objective jointly decide which downstream tasks are possible.

---

### Chapter 05 — Attention · 4 tasks · ★★–★★★

**Objective** Write scaled dot-product attention from the equation. If a student
does one task from this course properly, make it 5.1.

**Likely mistakes**

- *Softmax along the wrong axis.* Shapes still work, model still trains,
  computes something else. The self-check's row-sum assertion catches it.
- *Masking after the softmax* rather than before. Rows no longer sum to one.
- *Reaching for `nn.MultiheadAttention`.* The task says not to, and the check
  compares against PyTorch's kernel afterwards, which is the honest way to use it.

**Discussion** 5.4 asks whether a large attention weight is an explanation.
Expect over-confidence in heatmaps. The three reasons to land: the value vector
matters as much as the weight; residual connections route around attention
entirely; different distributions can give identical outputs (Jain & Wallace
2019). Then ask what ablation would produce *evidence*.

---

### Chapter 06 — Transformers · 4 tasks · ★★–★★★

**Objective** Assemble the architecture, and see what residuals buy.

**Likely mistakes**

- *Scaling by `sqrt(d_model)` instead of `sqrt(d_head)`* in 6.1.
- *`view` after `transpose` without `.contiguous()`*. PyTorch's error message is
  clear; the reason is not, so explain the memory layout.
- *Computing the positional divisor as a direct power* in 6.2. It underflows at
  large `d_model` and collapses the high-frequency dimensions.

**Discussion** 6.2 demonstrates permutation equivariance and then *breaks* it on
purpose. Flag forward to 10.1, where the same property is required rather than
removed. Same layer, opposite requirement, decided by the domain — this is the
cleanest example in the course of architecture following from the problem.

6.4 is another deliberate non-result. The Transformer does not clearly beat the
LSTM at this scale, and the honest conclusion is "no meaningful difference".

---

### Chapter 07 — Language Models · 5 tasks · ★★–★★★

**Objective** One causal mask and a vocabulary turn chapter 06 into GPT.

**Likely mistakes**

- *Shifting the target by more or less than one* in 7.1.
- *Not predicting the untrained loss before measuring it* in 7.2. Make them do
  it; `log(V)` catches initialisation bugs immediately.
- *Judging decoding settings by eye* in 7.4. The task requires a numeric
  diversity measure for a reason.

**Discussion** 7.5 — "just autocomplete". Require the strongest possible version
of *both* readings before a position. The answer that matters operationally is
that both readings agree fluency is uncorrelated with correctness, which turns an
unfalsifiable dispute into three measurable things.

---

### Chapter 08 — Pretrained LLMs and Adaptation · 4 tasks · ★★–★★★

**Objective** Stop training models, start adapting one, and choose the cheapest
point on the spectrum that works.

**Likely mistakes**

- *Mean pooling without the attention mask* (8.1). The bug the whole task
  exists for: nothing errors, short sentences are diluted by their batch-mates,
  accuracy quietly drops.
- *Skipping L2 normalisation.* Longer texts then dominate every similarity
  ranking for reasons unrelated to meaning.
- *Trusting `print_trainable_parameters()`* in 8.3 instead of deriving the count.
  The derivation is the task.

**Discussion** 8.2's TF-IDF baseline **matches** the embeddings, because the log
entries are templated. That is a finding about the data. Then 8.4 scenario 4 —
quarterly-updated grid codes — where the tempting answer is fine-tuning and the
right one is retrieval. *Fine-tune to change behaviour, retrieve to change
knowledge.*

---

### Chapter 09 — Foundation Models Beyond LLMs · 4 tasks · ★★–★★★

**Objective** A foundation model with no vocabulary and no tokens, run zero-shot
against specialists trained on the target data.

**Likely mistakes**

- *Building the harness after loading the model* (9.1). Fixing the protocol first
  is the point; the reference-scores-zero assertion is the cheap check.
- *Quantile crossing* going unnoticed in 9.2. The self-check asserts monotonicity.
- *Reporting coverage without sharpness* in 9.3. An infinitely wide interval has
  perfect coverage and no value.

**Discussion** 9.4(c): what the experiment does *not* support. One synthetic
series, one horizon, one cadence. Students will over-generalise; make them write
the three claims it cannot support. Also worth landing: "zero-shot" means no
gradient updates on this task, not that the model has never seen anything
similar — the synthetic data is the only reason the claim is clean here.

---

### Chapter 10 — Towards Grid Foundation Models · 6 tasks · ★★★

**Objective** Pretrain across grids, transfer to one never seen, and validate
against the physics rather than only the statistics.

**Likely mistakes**

- *Using `perm` where the inverse permutation belongs* in 10.1. Produces a
  different graph that looks plausible — same shapes, same degree distribution,
  wrong topology.
- *Mean instead of sum aggregation* in 10.2. Ask what makes a bus with two
  incident lines different from one with twenty; the physics is additive.
- *Masking the slack bus* in 10.3. Its voltage is the angular reference; the
  resulting loss term is pure noise.
- *Equalising fine-tuning budgets by epochs rather than steps* in 10.4. The
  label-efficiency curve then mostly measures the optimisation budget.
- *Fine-tuning in place across label budgets* instead of deep-copying, so each
  budget starts from the previous one's weights.

**Discussion** 10.5 is the chapter's argument: an excellent voltage MAE alongside
a power-balance mismatch of megawatts. The model has learned the *marginal
distribution* of voltages without the constraints that couple them. It is right
on average and wrong as a system state — the failure mode MAE cannot see, which
is why physics validation is reported separately and never averaged in.

**10.6 is the capstone and has no answer key.** Students design their own grid
foundation model against a ten-point template. Mark **internal consistency**, not
ambition: the commonest failure is a token choice that cannot support the claimed
downstream task, such as tokenising whole snapshots and then promising per-bus
anomaly localisation. The solution notebook gives one worked design and states
explicitly that it is not *the* answer; a student who argues convincingly that
branches beat buses as tokens has made a research contribution, not an error.

Good closing question for the course: *what would you have to measure to find
out which tokenisation is right?*

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

---

## Tutorial 11 — optional, and how to use it

Tutorial 11 (*Federated Learning and Federated Foundation Models*) is marked
**optional / advanced**. The ten-tutorial onboarding course is complete without
it, and the README says so. Nothing in 01–10 depends on it.

**Who it is for.** Readers who have finished the course and are choosing a
thesis topic. It ends with eleven open research questions, and they are open.

**Prerequisites.** Tutorials 04 (representation learning), 08 (LoRA) and 10
(the Mini-GridFM) specifically. A student who has not seen LoRA will not follow
§17, and one who has not seen masked pretraining will not follow §19.

**Runtime.** About 7 minutes in classroom mode, of which roughly 25 seconds is
Ray starting up for the Flower simulation. It is the heaviest notebook in the
course and has its own CI group for that reason.

**The extra dependency.** Tutorial 11 is the only notebook that needs
`flwr[simulation]`, which pulls in Ray. It is a normal dependency of the
project, so `uv sync` covers it; be aware it is the largest single install.

**Teaching notes.**

- §7 deliberately reports the *gradient-step budget* alongside the accuracy
  comparison. FedAvg spends the most local computation and still does not beat
  local training on this problem. That is the honest result, and students
  should be asked why before being told.
- §10 shows FedProx buying essentially nothing (3% drift reduction, worse
  accuracy). This is a negative result about the *benchmark*, not the method —
  three well-conditioned clients have no drift worth correcting. It is a good
  discussion prompt about when a method's published benefits transfer.
- §12 is the section to slow down on. The claim "federated learning makes data
  private" is common, wrong, and the exercise track's Task 11.6 asks students to
  dismantle it.
- §13 implements clipping and noise and **deliberately does not** implement an
  accountant or quote an epsilon. If a student asks what the epsilon is, the
  answer is that computing it is the missing step, and Opacus is where to get it.
- §20's transfer result — federated pretraining ahead at 5 labels, centralized
  ahead at 36 — is one seed on a tiny model. Present it as a methodology
  working, not a benchmark.
