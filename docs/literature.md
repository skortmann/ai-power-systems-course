# Literature and resources

Sources used to design and build this course. Every arXiv identifier below was
checked against the arXiv API in September 2026; titles and first-submission
dates are as returned by it. Where a work exists as both a preprint and a
journal article, the preprint is cited because it is what students can actually
open.

**How to read this list.** Prefer the primary sources. Blog posts and surveys
are marked as such and are included only where they genuinely add something —
usually a maintained implementation or an up-to-date benchmark.

---

## 0. The prompt's starting point, read critically

The course was commissioned with reference to a Medium article,
*"What are Foundation Models Beyond LLMs? Across Vision, Biology, Weather and
Robotics"* (mjgmario, medium.com). The article's framing — that foundation
models are a broader category than LLMs — is the right one and is the spine of
this course.

Two cautions were carried into the material rather than the framing:

1. Popular write-ups routinely use *pretrained*, *large*, *generative* and
   *foundation model* interchangeably. Tutorial 09 separates them explicitly
   (see `docs/foundation_models.md`).
2. Secondary sources age fast in this field and frequently mis-state parameter
   counts, dates and capabilities. Everything in this course that matters was
   re-checked against the primary paper or the model card.

The article itself was not reachable for direct quotation when the course was
built (HTTP 403 to automated fetches); it is credited as the framing prompt,
not as a technical source.

---

## 1. Foundations of machine learning and deep learning

These are textbook-level topics; the course cites standard references rather
than chasing priority.

- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd
  ed., Springer, 2009. Free PDF: <https://hastie.su.domains/ElemStatLearn/>
  — bias/variance, cross-validation, trees and boosting (tutorial 01).
- Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
- Goodfellow, Bengio & Courville, *Deep Learning*, MIT Press, 2016.
  <https://www.deeplearningbook.org/> — backpropagation, optimisation,
  regularisation (tutorial 02).
- Rumelhart, Hinton & Williams, "Learning representations by back-propagating
  errors", *Nature* 323, 1986. doi:10.1038/323533a0 — the paper that made
  backpropagation mainstream. (Earlier antecedents exist: Linnainmaa 1970,
  Werbos 1974. The course says so rather than picking a winner.)
- Rosenblatt, "The perceptron: a probabilistic model for information storage
  and organization in the brain", *Psychological Review* 65(6), 1958.
- Hochreiter & Schmidhuber, "Long Short-Term Memory", *Neural Computation* 9(8),
  1997. doi:10.1162/neco.1997.9.8.1735 — tutorial 03.
- LeCun, Bottou, Bengio & Haffner, "Gradient-based learning applied to document
  recognition", *Proc. IEEE* 86(11), 1998 — LeNet-5, CNNs.
- Krizhevsky, Sutskever & Hinton, "ImageNet Classification with Deep
  Convolutional Neural Networks", NeurIPS 2012 — AlexNet.
- Breiman, "Random Forests", *Machine Learning* 45, 2001.
- Friedman, "Greedy Function Approximation: A Gradient Boosting Machine",
  *Annals of Statistics* 29(5), 2001.

## 2. Representation learning and self-supervision

- Mikolov, Chen, Corrado & Dean, "Efficient Estimation of Word Representations
  in Vector Space", [arXiv:1301.3781](https://arxiv.org/abs/1301.3781) (2013-01-16)
  — Word2Vec; the origin of the "embeddings have geometry" intuition in
  tutorial 04.
- Bengio, Courville & Vincent, "Representation Learning: A Review and New
  Perspectives", [arXiv:1206.5538](https://arxiv.org/abs/1206.5538) (2012).
- Chen, Kornblith, Norouzi & Hinton, "A Simple Framework for Contrastive
  Learning of Visual Representations",
  [arXiv:2002.05709](https://arxiv.org/abs/2002.05709) (2020-02-13) — SimCLR,
  the NT-Xent loss implemented in `models/representation.py`.
- Grill et al., "Bootstrap your own latent",
  [arXiv:2006.07733](https://arxiv.org/abs/2006.07733) (2020-06-13) — BYOL;
  self-supervision without negatives.
- He, Chen, Xie, Li, Dollár & Girshick, "Masked Autoencoders Are Scalable
  Vision Learners", [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
  (2021-11-11) — the masking-ratio lesson transferred to load profiles in
  tutorial 04.

## 3. Attention, Transformers and language models

- Bahdanau, Cho & Bengio, "Neural Machine Translation by Jointly Learning to
  Align and Translate", [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
  (2014-09-01) — attention *before* the Transformer. Worth reading precisely
  because it separates the two ideas.
- Sutskever, Vinyals & Le, "Sequence to Sequence Learning with Neural
  Networks", [arXiv:1409.3215](https://arxiv.org/abs/1409.3215) (2014-09-10).
- Vaswani et al., "Attention Is All You Need",
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) (2017-06-12) — tutorials
  05 and 06 derive and rebuild this.
- Devlin, Chang, Lee & Toutanova, "BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding",
  [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) (2018-10-11).
- Radford et al., "Improving Language Understanding by Generative
  Pre-Training" (GPT-1), OpenAI, 2018; and "Language Models are Unsupervised
  Multitask Learners" (GPT-2), OpenAI, 2019.
- Brown et al., "Language Models are Few-Shot Learners",
  [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) (2020-05-28) — GPT-3 and
  the birth of in-context learning as a headline capability.
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified
  Text-to-Text Transformer", [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
  (2019-10-23) — T5; the encoder-decoder reference point, and the backbone
  Chronos was built on.
- Liu et al., "RoBERTa", [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
  (2019-07-26) — how much of BERT's result was the recipe, not the architecture.
- Dosovitskiy et al., "An Image is Worth 16x16 Words",
  [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) (2020-10-22) — ViT; the
  cleanest proof that Transformers are not about language.

### Scaling, tuning and adaptation

- Kaplan et al., "Scaling Laws for Neural Language Models",
  [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) (2020-01-23).
- Hoffmann et al., "Training Compute-Optimal Large Language Models",
  [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) (2022-03-29) — Chinchilla;
  corrected Kaplan's parameter/data trade-off. Cite both, in that order,
  whenever "scaling laws" comes up.
- Ouyang et al., "Training language models to follow instructions with human
  feedback", [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) (2022-03-04)
  — InstructGPT, RLHF.
- Rafailov et al., "Direct Preference Optimization",
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) (2023-05-29) — DPO.
- Schulman et al., "Proximal Policy Optimization Algorithms",
  [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) (2017-07-20).
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
  [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) (2021-06-17) — tutorial 08.
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs",
  [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) (2023-05-23).
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks", [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) (2020-05-22).
- Wei et al., "Chain-of-Thought Prompting",
  [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) (2022-01-28).
- Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
  [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) (2023-12-01) — the
  state-space alternative mentioned in tutorial 06's complexity discussion.

## 4. Foundation models as a category

- Bommasani et al., "On the Opportunities and Risks of Foundation Models",
  [arXiv:2108.07258](https://arxiv.org/abs/2108.07258) (2021-08-16) — the report
  that introduced the term. Tutorial 09 uses its definition and then stress-tests
  it. Note that the term was contested from the start; the course presents it as
  a useful label with fuzzy edges, not a crisp taxonomy.

### Vision

- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision",
  [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) (2023-04-14) — the best
  single counter-example to "foundation model = generative model": DINOv2
  generates nothing and is unambiguously a foundation model.
- Radford et al., "Learning Transferable Visual Models From Natural Language
  Supervision", [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
  (2021-02-26) — CLIP.
- Kirillov et al., "Segment Anything",
  [arXiv:2304.02643](https://arxiv.org/abs/2304.02643) (2023-04-05) — SAM;
  SAM 2 extends it to video (Ravi et al., 2024).

### Biology

- Jumper et al., "Highly accurate protein structure prediction with AlphaFold",
  *Nature* 596, 2021. doi:10.1038/s41586-021-03819-2. AlphaFold 3 (Abramson et
  al., *Nature* 630, 2024) extends the scope to complexes.
- Lin et al., "Evolutionary-scale prediction of atomic-level protein structure
  with a language model", *Science* 379(6637), 2023.
  doi:10.1126/science.ade2574 — ESM-2 / ESMFold. A *language* model whose
  language is amino acids: the clearest demonstration that the architecture is
  modality-agnostic.

### Weather and Earth systems

- Lam et al., "GraphCast: Learning skillful medium-range global weather
  forecasting", [arXiv:2212.12794](https://arxiv.org/abs/2212.12794)
  (2022-12-24) — a graph neural network on a physical domain; the closest
  methodological cousin to a GridFM in this list.
- Bodnar et al., "A Foundation Model for the Earth System" (Aurora),
  [arXiv:2405.13063](https://arxiv.org/abs/2405.13063) (2024-05-20).
- Price et al., "GenCast: Diffusion-based ensemble forecasting for medium-range
  weather", [arXiv:2312.15796](https://arxiv.org/abs/2312.15796) (2023-12-25).

### Time series

- Ansari et al., "Chronos: Learning the Language of Time Series",
  [arXiv:2403.07815](https://arxiv.org/abs/2403.07815) (2024-03-12).
- Ansari et al., "Chronos-2: From Univariate to Universal Forecasting",
  [arXiv:2510.15821](https://arxiv.org/abs/2510.15821) (2025-10-17) — 120 M
  parameters, encoder-only, univariate / multivariate / covariate-informed in
  one model, Apache-2.0. **This is the model tutorial 09 actually runs.** Model
  card: <https://huggingface.co/amazon/chronos-2>; code:
  <https://github.com/amazon-science/chronos-forecasting>.
- Das et al., "A decoder-only foundation model for time-series forecasting",
  [arXiv:2310.10688](https://arxiv.org/abs/2310.10688) (2023-10-14) — TimesFM.
- Woo et al., "Unified Training of Universal Time Series Forecasting
  Transformers", [arXiv:2402.02592](https://arxiv.org/abs/2402.02592)
  (2024-02-04) — Moirai.
- Goswami et al., "MOMENT: A Family of Open Time-series Foundation Models",
  [arXiv:2402.03885](https://arxiv.org/abs/2402.03885) (2024-02-06).
- Ekambaram et al., "Tiny Time Mixers (TTMs)",
  [arXiv:2401.03955](https://arxiv.org/abs/2401.03955) (2024-01-08) — the
  "small can be a foundation model too" data point, and a useful antidote to
  parameter-count worship.
- Garza & Mergenthaler-Canseco, "TimeGPT-1",
  [arXiv:2310.03589](https://arxiv.org/abs/2310.03589) (2023-10-05).
- Liu et al., "Moirai-MoE: Empowering Time Series Foundation Models with Sparse
  Mixture of Experts", [arXiv:2410.10469](https://arxiv.org/abs/2410.10469)
  (2024-10-14) — where mixture-of-experts enters the time-series story.

### Tabular

- Hollmann, Müller, Eggensperger & Hutter, "TabPFN: A Transformer That Solves
  Small Tabular Classification Problems in a Second",
  [arXiv:2207.01848](https://arxiv.org/abs/2207.01848) (2022-07-05) — a
  foundation model pretrained entirely on *synthetic* data, doing in-context
  learning on tables. Excellent for breaking the "foundation model = trained on
  the internet" assumption.

### Robotics

- Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to
  Robotic Control", [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)
  (2023-07-28).
- Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model",
  [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) (2024-06-13).
- Octo Model Team, "Octo: An Open-Source Generalist Robot Policy",
  [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) (2024-05-20).
- Black et al., "π0: A Vision-Language-Action Flow Model for General Robot
  Control", [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) (2024-10-31).

---

## 5. Foundation models for energy and power systems

This is the literature the capstone builds on. It is young, moves fast, and —
importantly for how the course frames it — consists almost entirely of research
demonstrations rather than operational systems.

### Position papers and roadmaps

- Hamann et al., "Foundation Models for the Electric Power Grid",
  [arXiv:2407.09434](https://arxiv.org/abs/2407.09434) (2024-07-12; published
  in *Joule*, 2024). **The reference point for tutorial 10.** Sets out the case
  for grid foundation models and sketches GridFM-v0 for power flow on graph
  neural networks.
- Huang et al., "Large Foundation Models for Power Systems",
  [arXiv:2312.07044](https://arxiv.org/abs/2312.07044) (2023-12-12) — mostly
  about applying *LLMs* to power-system tasks (OPF, EV scheduling, knowledge
  retrieval). Useful as a contrast: this is the narrower reading of "foundation
  model for power systems" that tutorial 09 argues against treating as the
  whole story.

### Models and implementations

- Tu et al., "PowerPM: Foundation Model for Power Systems",
  [arXiv:2408.04057](https://arxiv.org/abs/2408.04057) (2024-08-07; NeurIPS
  2024) — electricity time series with a temporal + hierarchical encoder,
  pretrained with masked modelling and dual-view contrastive learning,
  evaluated on a broad set of downstream tasks.
- **GridFM** (open source): `gridfm-graphkit` — training, fine-tuning and
  serving of grid foundation models, built on PyTorch Geometric and Lightning,
  with self-supervised masked-feature pretraining plus a physics-informed AC
  power-balance loss and zero-shot evaluation on unseen topologies.
  <https://github.com/gridfm/gridfm-graphkit>.
  The methodology of the Mini-GridFM in tutorial 10 deliberately mirrors this
  one (masked node features + physics loss + held-out grid), at roughly
  1/10000 of the scale.
- **GridFM DataKit** — companion dataset generator for power flow and OPF:
  <https://github.com/gridfm/gridfm-datakit>; described in
  [arXiv:2512.14658](https://arxiv.org/abs/2512.14658) (2025-12-16).
- **GridSFM**, Microsoft Research (2026) — a *small* foundation model for AC
  optimal power flow, trained across ~200 grids and ~500 k scenarios, producing
  bus voltages, dispatch, branch flows and a feasibility classification in
  milliseconds, and usable as a warm start for a conventional solver.
  <https://github.com/microsoft/gridSFM> and the Microsoft Research publication
  page. Note the name: *small* is doing deliberate work, and it is the best
  single citation against "foundation model = very large model".
- Fan et al., "WindFM: An Open-Source Foundation Model for Zero-Shot Wind
  Power Forecasting", [arXiv:2509.06311](https://arxiv.org/abs/2509.06311)
  (2025-09-08) — 8.1 M parameters, discretise-and-generate, decoder-only,
  pretrained on the WIND Toolkit (~126 000 sites). Reported to beat larger
  foundation models zero-shot. Another small-beats-large data point.
- Lin et al., "EnergyDiff: Universal Time-Series Energy Data Generation using
  Diffusion Models", [arXiv:2407.13538](https://arxiv.org/abs/2407.13538)
  (2024-07-18) — generative modelling of energy time series; relevant to the
  synthetic-data discussion in tutorial 09 and to `data/README.md`.
- "LUMINA: A Grid Foundation Model for Benchmarking AC Optimal Power Flow
  Surrogate Learning", [arXiv:2605.02133](https://arxiv.org/abs/2605.02133)
  (2026-05-04).

### Benchmarks and critical evaluations

These matter more than the model papers for the course's argument, because they
are where foundation models get compared against specialists on equal terms.

- "Empirical Assessment of Time-Series Foundation Models For Power System
  Forecasting Applications",
  [arXiv:2604.22077](https://arxiv.org/abs/2604.22077) (2026-04-23) — evaluates
  transformer and foundation models on PV, wind and load forecasting across
  several forecasting settings.
- "FETS Benchmark: Foundation Models Enable Scalable and Generalizable Energy
  Time Series Forecasting",
  [arXiv:2604.22328](https://arxiv.org/abs/2604.22328) (2026-04-24).
- Majumder et al., "Exploring the Capabilities and Limitations of Large
  Language Models in the Electric Energy Sector",
  [arXiv:2403.09125](https://arxiv.org/abs/2403.09125) (2024).

**Reading these three together is the assignment set at the end of tutorial 09.**
They do not all agree, and the disagreement is instructive: results depend
heavily on horizon, resolution, whether covariates are available, and how the
specialist baseline was tuned.

---

## 6. Data sources and licensing

- **Open Power System Data — Time series**, version 2020-10-06.
  <https://data.open-power-system-data.org/time_series/2020-10-06>,
  doi:10.25832/time_series/2020-10-06. Hourly load, wind, solar and day-ahead
  prices for Europe, 2015 to mid-2020, derived from ENTSO-E Transparency.
  Requested attribution:

  > Open Power System Data. 2020. Data Package Time series. Version 2020-10-06.
  > https://doi.org/10.25832/time_series/2020-10-06. (Primary data from various
  > sources, for a complete list see URL).

  OPSD's own data packages do not carry a single blanket open licence covering
  redistribution of the derived series, and the underlying ENTSO-E Transparency
  terms distinguish between datasets that may be freely re-used and those that
  may not. **This course therefore does not redistribute OPSD data.** It ships
  a clearly labelled synthetic dataset and provides
  `scripts/download_data.py`, which fetches and caches OPSD locally for
  students who want to repeat every experiment on real measurements. See
  `data/README.md`.

- **ENTSO-E Transparency Platform**, <https://transparency.entsoe.eu/> — the
  upstream source. Requires an account for API access, which is why the basic
  course does not depend on it.

- **SimBench**, <https://simbench.de/> — realistic German MV/LV benchmark
  networks with time series. Optional extra (`uv sync --extra simbench`); the
  Python package is ~90 MB, which is why it is not in the base install.
  Meinecke et al., "SimBench — A Benchmark Dataset of Electric Power Systems to
  Compare Innovative Solutions Based on Power Flow Analysis", *Energies* 13(12),
  2020. doi:10.3390/en13123290

- **pandapower test networks** — the IEEE and MATPOWER cases used in tutorial
  10 ship with pandapower itself, so nothing is downloaded.
  Thurner et al., "pandapower — An Open-Source Python Tool for Convenient
  Modeling, Analysis, and Optimization of Electric Power Systems",
  *IEEE Transactions on Power Systems* 33(6), 2018.
  doi:10.1109/TPWRS.2018.2829021

- **NREL WIND Toolkit**, <https://www.nrel.gov/grid/wind-toolkit.html> — the
  pretraining corpus behind WindFM; mentioned, not used.

---

## 7. Software

- PyTorch — Paszke et al., NeurIPS 2019. <https://pytorch.org/>
- scikit-learn — Pedregosa et al., JMLR 12, 2011. <https://scikit-learn.org/>
- Hugging Face `transformers` — Wolf et al., EMNLP 2020 demos.
  <https://github.com/huggingface/transformers>
- Hugging Face `peft` — <https://github.com/huggingface/peft>
- `chronos-forecasting` — <https://github.com/amazon-science/chronos-forecasting>
- pandapower — <https://www.pandapower.org/>
- PyTorch Geometric — Fey & Lenssen, 2019. <https://pyg.org/>. *Not* a
  dependency of this course: tutorial 10 implements message passing with
  `index_add_` in about ten lines so the scatter-add is visible. PyG is the
  right choice for real work and is signposted as such.

---

## 8. Evaluation, uncertainty and responsible practice

- Gneiting & Raftery, "Strictly Proper Scoring Rules, Prediction, and
  Estimation", *JASA* 102(477), 2007 — the theory behind CRPS and pinball loss.
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed.,
  OTexts. <https://otexts.com/fpp3/> — the source of the course's insistence on
  seasonal-naive baselines.
- Makridakis, Spiliotis & Assimakopoulos, "The M4 Competition: 100,000 time
  series and 61 forecasting methods", *International Journal of Forecasting*
  36(1), 2020 — the empirical record on simple methods beating complex ones.
- Kapoor & Narayanan, "Leakage and the Reproducibility Crisis in ML-based
  Science", *Patterns* 4(9), 2023.
  [arXiv:2207.07048](https://arxiv.org/abs/2207.07048) — the data-leakage
  taxonomy behind tutorial 01's leakage section.
- Mitchell et al., "Model Cards for Model Reporting",
  [arXiv:1810.03993](https://arxiv.org/abs/1810.03993) (2018).
- Gebru et al., "Datasheets for Datasets",
  [arXiv:1803.09010](https://arxiv.org/abs/1803.09010) (2018).
- Strubell, Ganesh & McCallum, "Energy and Policy Considerations for Deep
  Learning in NLP", [arXiv:1906.02243](https://arxiv.org/abs/1906.02243) (2019).
- Sculley et al., "Hidden Technical Debt in Machine Learning Systems",
  NeurIPS 2015.
