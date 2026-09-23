# What is a foundation model?

The reference document for the course's central claim. Tutorial 09 builds the
argument experimentally; this is the written version, and it is where the
misconceptions listed in the README are dealt with properly.

---

## 1. The definition, and what it actually requires

> A **foundation model** is a model trained on broad data at scale, designed to
> be adapted to a wide range of downstream tasks.
> — after Bommasani et al., *On the Opportunities and Risks of Foundation
> Models* ([arXiv:2108.07258](https://arxiv.org/abs/2108.07258), 2021)

Three requirements, all load-bearing:

1. **Broad pretraining data.** Not one dataset, one plant, one grid. Breadth is
   what makes the learned representation general rather than a fit to one
   distribution.
2. **A self-supervised (or otherwise task-agnostic) objective.** The model must
   be trainable without task labels, because task labels are what limit scale.
3. **Adaptability to many downstream tasks**, via zero-shot use, prompting,
   probing, PEFT or fine-tuning.

Note what is *not* in the definition: size, generativity, language, and
Transformers. All four are common in practice and none is required.

The term was contested from the day it was coined — critics found it
simultaneously over-claiming ("foundation") and under-specified. This course
uses it because the literature settled on it, and tightens it rather than
assuming it.

---

## 2. Five things that are routinely confused

| | definition | a clear example | a clear counter-example |
|---|---|---|---|
| **pretrained model** | trained before you got it | an ImageNet ResNet-50 | — |
| **reusable backbone** | its representation transfers usefully | a ResNet feature extractor | a model whose features only work for its own task |
| **foundation model** | broad pretraining + broad adaptability | DINOv2, Chronos-2, TabPFN | a 70B model fine-tuned to do exactly one thing |
| **large language model** | a large model over text | GPT-4, Llama-3-70B | DINOv2 (no language), TabPFN (no language) |
| **generative model** | samples from a data distribution | Stable Diffusion, GPT | DINOv2, TabPFN, most encoders |

The containment relations:

- Every foundation model is a pretrained model. **The converse is false.**
- Most current LLMs are foundation models. **Most foundation models are not
  LLMs.**
- Some foundation models are generative; many are not.

The cleanest single counter-example to remember is **DINOv2**: it generates
nothing, understands no language, and is unambiguously a foundation model. The
second-cleanest is **TabPFN**: pretrained entirely on *synthetic* data, tiny by
LLM standards, and it does in-context learning on tables.

---

## 3. Nine misconceptions

These are taught explicitly because students arrive with them.

### 1. "Foundation model = LLM." **False.**
Language is one modality. Vision (DINOv2, CLIP, SAM), biology (ESM, AlphaFold),
weather (GraphCast, Aurora, GenCast), time series (Chronos-2, TimesFM, Moirai),
tabular data (TabPFN), robotics (RT-2, OpenVLA, Octo, π0) and now power grids
(GridFM, PowerPM, GridSFM, WindFM) all have them.

### 2. "Foundation model = very large model." **Not necessarily.**
WindFM has 8.1 million parameters and beats larger models zero-shot on wind
power. TinyTimeMixers are in the same range. Microsoft named its grid model
**GridSFM** — *Small* Foundation Model — on purpose. Size correlates with
capability within a family; it is not the defining property.

### 3. "Transformer = LLM." **False.**
The Transformer is an architecture over sets of vectors plus positional
information. Tutorial 06 trains one that never sees a word. Meanwhile some
foundation models are not Transformers at all.

### 4. "Pretrained model = foundation model." **Not necessarily.**
An ImageNet classifier is pretrained. It was trained on one dataset with one
supervised objective for one task family. It transfers usefully — it is a good
backbone — but it is not designed for broad adaptation.

### 5. "Foundation models always beat specialist models." **False.**
Tutorial 09 measures a case where the zero-shot foundation model *loses* to a
small specialist, and the reasons generalise: a specialist trained on your
distribution, with your covariates, at your resolution, has advantages no
general model can assume. The foundation model's win is that it needs no
training data from you at all.

### 6. "RAG trains the model." **False.**
Not one weight changes. Retrieved text enters through the context window exactly
like anything you type. RAG is a system architecture around a frozen model.

### 7. "Attention explains why the model decided." **Usually not.**
An attention weight is not a contribution: the value vector matters too,
residual connections route around attention entirely, and different attention
patterns can produce identical outputs. Tutorial 05 ablates the weights and
reports a number instead of narrating a heatmap.

### 8. "Higher accuracy implies physical correctness." **False.**
Tutorial 01 has a PV forecaster with a respectable $R^2$ that predicts negative
generation a quarter of the time and non-zero output at midnight in *every*
night hour. No statistical metric catches this.

### 9. "Zero-shot means the model has never seen anything similar." **False.**
It means no examples of this task were provided *now*. What was in the
pretraining corpus is usually unknown and often unknowable. Public benchmarks
are frequently contaminated. This is why tutorial 09 evaluates on a dataset that
provably could not have been pretrained on — the synthetic series is generated
by this repository.

---

## 4. Foundation models by domain

| domain | representative models | input representation | pretraining objective | architecture | downstream tasks |
|---|---|---|---|---|---|
| language / code | GPT, Llama, Mistral, Qwen | subword tokens | next-token prediction | decoder Transformer | generation, QA, extraction, code |
| vision (self-sup.) | DINOv2 | image patches | self-distillation, no labels | ViT | classification, segmentation, depth, retrieval |
| vision-language | CLIP | image + text pairs | contrastive alignment | dual encoder | zero-shot classification, retrieval |
| segmentation | SAM, SAM 2 | image/video + prompt | supervised on a huge auto-labelled set | ViT + prompt decoder | promptable segmentation |
| biology (sequence) | ESM-2 | amino-acid tokens | masked language modelling | encoder Transformer | structure, function, variant effects |
| biology (structure) | AlphaFold 2/3 | sequence + MSA + templates | supervised on the PDB | Evoformer + structure module | structure prediction, complexes |
| weather | GraphCast | gridded atmospheric state | next-state prediction | graph neural network | medium-range forecasting |
| weather | Aurora, GenCast | gridded state | masked / diffusion | 3D Swin, diffusion | multi-variable and ensemble forecasts |
| time series | Chronos-2 | numeric context (+ covariates) | quantile forecasting over broad corpora | encoder Transformer, group attention | zero-shot univariate / multivariate forecasting |
| time series | TimesFM, Moirai, MOMENT, TTM | patches of steps | next-patch / masked | decoder or encoder | forecasting, imputation, anomaly detection |
| tabular | TabPFN | a whole small table | trained on synthetic datasets | encoder Transformer | in-context classification/regression |
| robotics | RT-2, OpenVLA, Octo, π0 | images + language + proprioception | action prediction from demonstrations | VLM + action head, flow matching | manipulation, generalist control |
| **power grids** | GridFM, PowerPM, GridSFM, WindFM | graphs, ETS, operating states | masked reconstruction + physics loss | graph Transformer / GNN / decoder | power flow, OPF, forecasting, contingency |

Read the column "pretraining objective" downwards. Almost all of it is
"reconstruct something that was hidden" or "predict the next thing". That is the
whole trick, applied to different data.

---

## 5. Why power systems are hard for this recipe

Everything in the table above operates on data with a **fixed, shared
structure**: images are grids of pixels, text is a sequence of tokens, proteins
are sequences of residues. A power grid is not.

1. **Grid A and Grid B do not have the same number of buses**, the same
   topology, the same voltage levels or the same equipment. A fixed-width model
   trained on a 30-bus network cannot even be *evaluated* on a 118-bus one.
   Message passing and graph attention solve this, which is why every serious
   GridFM proposal is graph-based.
2. **The data is genuinely multimodal**: time series (P, Q, V, I, f, PMU, SCADA,
   smart meters), graph structure (buses, branches, switches, transformers),
   static metadata (voltage levels, conductor parameters, geography), weather,
   market signals, and text (logs, disturbance reports, standards).
3. **Physics is not optional.** A statistically excellent state that violates
   Kirchhoff's laws is worthless. This is an advantage as well as a constraint:
   the physics is a free, exact source of supervision, which is why
   physics-informed losses appear in essentially every GridFM implementation.
4. **Labels are expensive but simulation is cheap.** Unlike language, we can
   *generate* unlimited correct data with a power-flow solver. GridSFM used
   ~500,000 scenarios across ~200 grids; tutorial 10 uses a few thousand across
   seven. This makes the field's data situation unusual and rather favourable.
5. **The deployment bar is much higher.** A hallucinating chatbot is
   embarrassing. A hallucinating state estimator is a safety incident. Nothing
   in a notebook benchmark licenses operational use.

---

## 6. How to tell whether something deserves the name

A checklist, usable on any paper or product claiming to be a foundation model:

- [ ] **Breadth.** How many distinct sources/systems/datasets in pretraining?
      One dataset is not breadth, however large it is.
- [ ] **Task-agnostic objective.** Was it pretrained without the downstream
      labels? If the pretraining objective *is* the downstream task, it is a
      specialist.
- [ ] **Adaptation demonstrated.** Are there results on tasks that were not
      targets of pretraining — and ideally on *systems* not seen in pretraining?
- [ ] **Transfer to unseen structure.** For grids specifically: was it evaluated
      on a topology held out entirely, or only on new operating points of the
      same networks? These are very different claims.
- [ ] **Honest baselines.** Is it compared against a well-tuned specialist and
      a trivial baseline, or only against other large models?
- [ ] **Contamination addressed.** Could the evaluation data have been in
      pretraining? If the authors do not discuss this, assume it could.
- [ ] **Cost reported.** Parameters, memory, latency, and the compute for both
      pretraining and adaptation.

A model that fails the first three is a pretrained model, which is a perfectly
respectable thing to be. It is just a different claim.

---

## 7. The question the course ends on

> **What should the "tokens" of an electrical grid foundation model be?**

Candidate answers, none settled:

| candidate | argument for | argument against |
|---|---|---|
| **time steps** | matches time-series FMs directly | ignores topology entirely |
| **measurements** | finest granularity, handles missing data naturally | enormous sequences, no structure |
| **buses** | natural graph nodes; what tutorial 10 uses | a bus is not a fixed-size object across voltage levels |
| **branches** | flows are what constraints are written on | loses nodal balance structure |
| **operating states** | one token per snapshot; matches contingency work | throws away spatial detail |
| **subgraphs / zones** | matches how operators think; controls sequence length | how do you choose the partition? |
| **events** | sparse, meaningful, aligned with operator reasoning | requires labelled event detection |
| **learned latent tokens** | let the model decide (à la Perceiver) | uninterpretable; hard to validate physically |

And behind it, the question that actually matters:

> **What must a model learn so that knowledge acquired on Grid A remains useful
> on previously unseen Grid B?**

Tutorial 10's answer, which is the same answer tutorial 01 gives in miniature:
the representation has to be in units that mean the same thing everywhere
(per-unit, not volts), about relationships rather than identities (message
passing, not bus indices), and constrained by something that is true in both
grids (the power-flow equations). That is not a foundation-model insight. It is
just good modelling — which is rather the point.

---

## A second axis: where the pretraining data lives

*Tutorial 11 (optional/advanced).*

Everything above assumes the pretraining corpus can be assembled. For grids
that assumption is doing a lot of work. There is no public corpus of feeder
models and measurements, and the reason is not technical — it is privacy law,
commercial confidentiality, and critical-infrastructure rules. The data exists;
it is held by operators who cannot hand it over.

That makes *federated* pretraining more than an efficiency question for this
domain specifically. For language, federation is one option among several,
because a public corpus exists. For grids, it may be the only route to breadth
of pretraining at all.

Four lifecycle arrangements are worth distinguishing, because "federated
foundation model" is used for all of them:

| | Pretraining | Adaptation | When it applies |
|---|---|---|---|
| **A** | federated, on private data | central | no public corpus exists |
| **B** | central, on public data | federated on private data | a good general model already exists |
| **C** | central | local adapters, never aggregated | participants want no coupling |
| **D** | federated | federated | strictest, most expensive |

For power systems **B** is the pragmatic near-term option and **A** is the open
research question.

Two constraints shape everything in that column.

**Arithmetic.** A billion parameters at float32 is 4 GB per message. A hundred
clients over a hundred rounds is petabytes. Federating a foundation model the
naive way is not expensive, it is impossible — which makes parameter-efficient
adaptation a precondition rather than an optimisation.

**Language.** Federation keeps raw training samples local. It does not make
them private: updates leave every round and are a deterministic function of the
data. Secure aggregation and differential privacy address different parts of
what remains, and neither is implied by the word "federated". Tutorial 11 keeps
that distinction explicit throughout, and says plainly where it implements the
arithmetic of a mechanism without implementing the guarantee.
