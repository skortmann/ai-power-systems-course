# From Machine Learning to Foundation Models

### Hands-On Artificial Intelligence for Power & Energy Systems

A ten-notebook course that walks from linear regression to grid foundation
models, building every idea from scratch once and then using the production
library for it — with electrical power systems as the running application
throughout.

```
Task-specific ML  →  Neural networks  →  Representation learning  →  Attention
      →  Transformers  →  Large-scale pretraining  →  Foundation models
            →  Domain foundation models  →  Grid Foundation Models
```

---

## What this course is actually about

Most AI courses end at "and then there are LLMs". This one makes a narrower and
more useful argument:

> **Foundation models are a broader paradigm than LLMs.** The transition that
> matters is from training *one model for one dataset and one task* towards
> *pretraining reusable representations over broad distributions of data* that
> can then be adapted, prompted, fine-tuned or transferred to many downstream
> tasks.

Language models are one column of that table. Vision, biology, weather, time
series, tabular data, robotics — and, since about 2024, electrical grids — are
the others. By tutorial 10 you will have pretrained a small foundation model
across several power networks and transferred it to a grid it has never seen.

Two commitments run through every notebook:

1. **Nothing is a black box.** Attention is derived in NumPy before
   `nn.MultiheadAttention` appears. Backpropagation is hand-written and checked
   against finite differences. Message passing is ten lines of `index_add_`.
2. **The prose matches the measurement.** Where an experiment contradicted the
   expected story — and it did, several times — the text was rewritten, not the
   experiment. Several notebooks report a hypothesis that *failed*.

---

## Who it is for

New Bachelor's and Master's students, research assistants and early-stage PhD
students joining a power-systems research group.

**Prerequisites:** Python (loops, functions, NumPy arrays), first-year linear
algebra and calculus, and enough electrical engineering to know what a bus, a
line and a power flow are. No machine-learning background is assumed. Tutorial
10 is easier if you have met `pandapower`, but it does not require it.

**Not required:** a GPU. Everything runs on a laptop CPU. Optional sections that
benefit from a GPU are marked as such and are never needed to finish a tutorial.

---

## Install and run

```bash
git clone <this-repository>
cd ai-power-systems-course

uv sync                    # Python 3.12, ~5 minutes, CPU-only PyTorch
uv run jupyter lab         # then open tutorials/01_classical_machine_learning.ipynb
```

That is the whole setup. [`uv`](https://docs.astral.sh/uv/) handles the Python
version and the lockfile; there is no conda environment to manage.

<details>
<summary>Optional extras</summary>

```bash
uv sync --extra simbench   # realistic German MV/LV benchmark grids (~90 MB)
uv run python scripts/download_data.py   # the real OPSD measurements
```

**GPU users:** the lockfile pins the CPU build of PyTorch so the install stays
small. To use CUDA, install the matching wheel over it:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Nothing else changes; every notebook calls `torch_device()`.
</details>

**Start with tutorial 01** and go in order. Each notebook builds on the previous
one, and tutorials 01–09 all forecast the same series so the models are directly
comparable — there is a shared leaderboard that fills up as you go.

---

## The ten tutorials

| # | AI concept | Power-system application | Main library | ~Runtime |
|---|---|---|---|---|
| [01](tutorials/01_classical_machine_learning.ipynb) | Supervised learning, splits, leakage, bias–variance, trees | Day-ahead load forecasting; physical validity of a PV forecast | scikit-learn | 30 s |
| [02](tutorials/02_neural_networks.ipynb) | Neurons, backprop from scratch, autograd, optimisation | The same forecast, learned representation | NumPy → PyTorch | 45 s |
| [03](tutorials/03_rnns_and_lstms.ipynb) | Sequences, RNNs, vanishing gradients, LSTM | Load forecasting from a raw 168-hour window | PyTorch | 4 min |
| [04](tutorials/04_representation_learning.ipynb) | Autoencoders, masking, contrastive learning, label efficiency | Estimating hidden PV from net-load profiles | PyTorch | 45 s |
| [05](tutorials/05_attention.ipynb) | Q/K/V, scaled dot product, masks, multi-head | What a forecaster attends to in a load series | NumPy → PyTorch | 30 s |
| [06](tutorials/06_transformers.ipynb) | Blocks, residuals, LayerNorm, encoder/decoder, $O(n^2)$ | Transformer forecasting vs. the LSTM, head to head | PyTorch | 18 min |
| [07](tutorials/07_language_models.ipynb) | Tokenization, next-token prediction, sampling, scaling laws | A tiny GPT trained on power-engineering text | PyTorch | 7 min |
| [08](tutorials/08_pretrained_llms_and_adaptation.ipynb) | Transfer, frozen backbones, LoRA/PEFT, in-context learning, RAG | Classifying operator events; retrieval over a technical handbook | Transformers, PEFT | 3 min |
| [09](tutorials/09_foundation_models_beyond_llms.ipynb) | **What a foundation model actually is**; zero-shot transfer | Chronos-2 zero-shot vs. trained specialists | chronos-forecasting | 3 min |
| [10](tutorials/10_grid_foundation_models.ipynb) | Graph networks, multi-grid pretraining, physics-informed losses | **Mini-GridFM**: pretrain across grids, transfer to an unseen one | pandapower, PyTorch | 4 min |

Runtimes are wall-clock on a four-thread laptop CPU, measured on the committed
build. Tutorial 06 is the outlier because it trains four models; its section 10
explains why the Transformer is the slow one here. Set `AI_POWER_COURSE_FAST=1`
for the reduced configuration CI uses, which runs every notebook in well under a
minute.

### The power-system thread

The domain evolves alongside the method rather than being bolted on at the end:

```
01  a task-specific load predictor
02  a neural load predictor
03  a temporal load model
04  a self-supervised energy representation
05  attention over power-system time series
06  a Transformer forecasting model
07  generative modelling of engineering text
08  reusing a pretrained LLM for engineering text
09  reusing a pretrained time-series foundation model
10  pretraining reusable representations across electrical grids
```

---

## What you will be able to do afterwards

- Construct leak-free splits, and recognise leakage when you see it (tutorial 01
  measures a 30% fake improvement from one shuffled split).
- Implement backpropagation, attention, a Transformer, a GPT and a graph neural
  network, and check each against the library version.
- Explain supervised, unsupervised and self-supervised learning, and why the
  third one changed everything.
- Distinguish a pretrained model, a reusable backbone, a foundation model, an
  LLM and a generative model — and say which of the nine common misconceptions
  in [`docs/foundation_models.md`](docs/foundation_models.md) you used to hold.
- Use LoRA, build a RAG system, and explain precisely what each does *not* do.
- Evaluate a model on accuracy, calibration **and physical plausibility**, and
  explain why the third is a separate question.
- Read a 2026 AI-for-energy paper without treating the architecture as magic.

---

## Nine things the course will talk you out of

| | |
|---|---|
| "Foundation model = LLM" | Language is one modality of many |
| "Foundation model = very large model" | WindFM: 8.1 M parameters. GridSFM: *S* for Small |
| "Transformer = LLM" | Tutorial 06 trains one that never sees a word |
| "Pretrained = foundation model" | Breadth and adaptability, not provenance |
| "Foundation models beat specialists" | Tutorial 09 measures a case where it loses |
| "RAG trains the model" | Not one weight changes |
| "Attention explains the decision" | Tutorial 05 ablates the weights and reports a number |
| "Better accuracy = physically correct" | Tutorial 01: good $R^2$, negative PV at midnight |
| "Zero-shot = never seen anything similar" | Contamination is real and usually unmeasurable |

---

## Repository layout

```
├── tutorials/              the ten notebooks (committed with outputs)
│   └── _sources/           their jupytext sources — edit these, not the .ipynb
├── src/ai_power_course/    the reusable package
│   ├── data.py             the course dataset, with provenance attached
│   ├── synthetic.py        the physically motivated data generator
│   ├── corpus.py           the power-engineering text corpus
│   ├── metrics.py          point, probabilistic and *physical* metrics
│   ├── diagrams.py         every explanatory figure, as code
│   ├── models/             baselines, MLP/RNN/LSTM/Transformer, attention, GPT, GNN
│   └── grid/               pandapower networks, sampling, graphs, physics checks
├── data/                   see data/README.md for provenance and licensing
├── scripts/                data download, grid generation, notebook build
├── tests/                  ~140 tests, including leakage and physics checks
└── docs/                   literature, timeline, glossary, foundation models,
                            course overview, instructor guide
```

### Documentation

| file | what it is for |
|---|---|
| [`docs/course_overview.md`](docs/course_overview.md) | the design, the narrative arc, what is deliberately omitted |
| [`docs/foundation_models.md`](docs/foundation_models.md) | the conceptual reference: definitions, misconceptions, the domain table |
| [`docs/literature.md`](docs/literature.md) | every source, with arXiv IDs verified against the arXiv API |
| [`docs/ai_timeline.md`](docs/ai_timeline.md) | 1943 → 2026, with contested priorities flagged |
| [`docs/glossary.md`](docs/glossary.md) | precise definitions, pointing at where each idea is built |
| [`docs/instructor_guide.md`](docs/instructor_guide.md) | timings, likely difficulties, discussion questions, expected outcomes |
| [`data/README.md`](data/README.md) | why the shipped data is synthetic and how to get the real thing |

---

## Data, honestly

The committed dataset is **simulated, not measured**. It is generated from solar
geometry, a turbine power curve, a temperature-driven load model and a
merit-order price stack, with magnitudes tuned to resemble Germany (479 TWh/a,
PV capacity factor 0.12, wind 0.20). Every notebook prints a provenance banner
saying so.

It is synthetic because the obvious real alternative — Open Power System Data's
ENTSO-E-derived series — does not carry terms that clearly permit
redistribution. One command downloads it under the publisher's own terms:

```bash
uv run python scripts/download_data.py
```

after which every experiment runs unchanged on real measurements. Doing that and
checking which conclusions survive is exercise 3 of tutorial 01, and it is the
most instructive exercise in the notebook.

---

## Reproducibility

- One seed (`ai_power_course.config.SEED`), set before every model.
- Chronological splits everywhere; the test set is touched once.
- Dependencies pinned in `uv.lock`.
- No model weights in the repository; pretrained models download from their
  hubs on first use.
- CI executes **all ten notebooks from a clean install** in reduced mode, plus
  the test suite.

```bash
uv run pytest                                      # the test suite
uv run python scripts/build_notebooks.py           # rebuild every notebook
# a subset, or the CI configuration, without editing the file:
uv run python -c "import sys; sys.path.insert(0,'scripts'); \
  from build_notebooks import build; build(only=('03_',), fast=True)"
uv run ruff check .                                # lint
```

Notebooks are authored as [jupytext](https://jupytext.readthedocs.io/)
percent-format `.py` files under `tutorials/_sources/` and built into `.ipynb`
with outputs. **Edit the source, not the notebook.**

---

## Citation

If this course is useful in teaching or research, please cite it via
[`CITATION.cff`](CITATION.cff). If you use the data, cite the upstream source as
well — see [`data/README.md`](data/README.md).

## Licence

MIT, for the code, the notebooks and the generated data. See
[`LICENSE`](LICENSE). Third-party models downloaded at runtime carry their own
licences (all Apache-2.0 as configured); third-party data carries the terms of
its publisher.

---

## A closing note on how to use this

The exercises are not API drills. They ask you to explain why a baseline beat a
Transformer, to transfer an encoder to a topology excluded from pretraining, to
decide whether a foundation model's zero-shot win survives a distribution shift.
Several have no settled answer — the last one in the course is a genuinely open
research question:

> **What should the "tokens" of an electrical grid foundation model be?**

If you finish tutorial 10 with a defensible opinion about that, the course has
done its job.
