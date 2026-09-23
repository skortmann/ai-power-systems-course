# A timeline from the perceptron to grid foundation models

Dates are the **first public appearance** — the arXiv preprint where one exists,
otherwise the conference or journal publication. Several entries are genuinely
contested and are marked as such: priority disputes in this field are real, and
a teaching timeline should not quietly pick a winner.

The figure version of this is `ai_power_course.diagrams.ai_timeline()`, which is
deliberately split into two panels because a single linear axis makes everything
after 2015 unreadable — and that unreadability is itself the point.

---

## 1943–1990: the long build-up

| year | event | note |
|---|---|---|
| 1943 | McCulloch & Pitts formal neuron | a logical abstraction, not a learning rule |
| 1957/58 | **Perceptron** (Rosenblatt) | first learning algorithm for a neuron |
| 1969 | Minsky & Papert, *Perceptrons* | showed a single layer cannot represent XOR |
| 1970 | Linnainmaa: reverse-mode automatic differentiation | backpropagation, before the name |
| 1974 | Werbos applies it to neural networks | |
| 1980 | Fukushima's Neocognitron | convolutional structure, before backprop training |
| 1986 | **Rumelhart, Hinton & Williams** | the paper that made backpropagation mainstream |
| 1989 | Universal approximation theorems (Cybenko, Hornik) | expressiveness, not trainability |

> **Contested:** "who invented backpropagation" has at least four defensible
> answers. The course cites 1986 as the point it became *standard practice* and
> names the antecedents.

## 1990–2011: statistics wins

| year | event |
|---|---|
| 1995 | Support vector machines (Cortes & Vapnik) |
| **1997** | **LSTM** (Hochreiter & Schmidhuber) — the fix for vanishing gradients |
| **1998** | **LeNet-5** (LeCun et al.) — CNNs on real data |
| 2001 | Random forests (Breiman); gradient boosting (Friedman) |
| 2006 | Deep belief networks (Hinton et al.) — the "deep learning" label sticks |
| 2009 | ImageNet released — the dataset that made the next decade possible |

Through this period, structured/tabular problems were won by ensembles of trees,
and they largely still are (tutorials 01 and 02 reproduce that result).

## 2012–2016: deep learning takes over perception

| year | event |
|---|---|
| **2012** | **AlexNet** wins ImageNet by a wide margin |
| **2013** | **Word2Vec** (Mikolov et al.) — representations with usable geometry |
| 2014 | **Seq2seq** (Sutskever et al.); GANs (Goodfellow et al.); Adam (Kingma & Ba) |
| **2014/15** | **Attention** (Bahdanau, Cho & Bengio) — inside an RNN, for translation |
| 2015 | ResNet (He et al.) — residual connections make depth trainable |
| 2016 | AlphaGo |

> Note the three-year gap between attention (2014) and the Transformer (2017).
> Attention is older than the Transformer, and neither is a language model.

## 2017–2020: the architecture, then the scale

| year | event |
|---|---|
| **2017** | **Transformer** — *Attention Is All You Need* (Vaswani et al.) |
| **2018** | **BERT** (encoder) and **GPT-1** (decoder) split it in half |
| 2019 | GPT-2; RoBERTa; T5 |
| **2020** | **GPT-3** — in-context learning as a headline capability |
| 2020 | **Kaplan et al.** scaling laws; **Vision Transformer**; AlphaFold 2 at CASP14; SimCLR |
| 2020 | **RAG** (Lewis et al.) |

## 2021–2023: "foundation model" and the open ecosystem

| year | event |
|---|---|
| **2021** | **CLIP**; **"On the Opportunities and Risks of Foundation Models"** coins the term |
| 2021 | **LoRA** (Hu et al.); AlphaFold 2 published in *Nature* |
| **2022** | **InstructGPT** / RLHF; **Chinchilla** corrects the scaling trade-off; Stable Diffusion; ChatGPT (November) |
| 2022 | ESM-2 / ESMFold; **GraphCast** (December) |
| **2023** | LLaMA and the open-weight wave; **SAM**; **DINOv2**; QLoRA; DPO; Mamba; GenCast; RT-2 |

> **Contested:** the term *foundation model* was disputed from the moment it was
> coined — critics argued it over-claimed and under-specified. This course uses
> it because it is the term the literature settled on, and spends tutorial 09
> tightening the definition rather than assuming it.

## 2024–2026: domain-specific foundation models

| year | event |
|---|---|
| **2024** | **Chronos**, **TimesFM**, **Moirai**, **MOMENT**, **TinyTimeMixers** — time-series FMs arrive in force |
| 2024 | **Aurora** (Earth system); AlphaFold 3; OpenVLA, Octo, π0 (robotics) |
| **2024-07** | **"Foundation Models for the Electric Power Grid"** (Hamann et al., *Joule*) — the GridFM agenda |
| 2024-08 | **PowerPM** (NeurIPS 2024) — electricity time series, masked + contrastive pretraining |
| **2025** | **WindFM** (8.1 M parameters, zero-shot wind power); `gridfm-graphkit` and `gridfm-datakit` released |
| 2025-10 | **Chronos-2** — 120 M parameters, univariate / multivariate / covariate-informed in one encoder |
| **2026** | **GridSFM** (Microsoft Research) — a *small* foundation model for AC-OPF across ~200 grids |
| 2026 | LUMINA (AC-OPF surrogate benchmark); FETS and other energy-forecasting FM benchmarks |

---

## What the shape of this timeline is trying to teach

**The ideas are old; the scale is new.** Backpropagation is from the 1970s–80s,
convolutions from 1980, LSTMs from 1997, attention from 2014. What changed after
2012 was data, hardware and engineering.

**Architecture and modality are independent.** The 2017 Transformer was a
translation model. By 2020 the same block was doing images, by 2022 proteins and
weather, by 2024 time series and robot actions, by 2026 power grids. Anyone who
believes "Transformer = LLM" cannot make sense of the last five rows.

**Scale drove capability, then stopped being the only story.** 2020–2022 was
about making models bigger. 2023 onwards is increasingly about making them
*narrower and better suited*: TinyTimeMixers, WindFM at 8 M parameters, GridSFM
explicitly named "small". The foundation-model property is breadth of
pretraining and adaptability, not parameter count.

**Energy is a late adopter, and that is an opportunity.** The gap between "the
Transformer exists" (2017) and "there is a serious foundation-model agenda for
power grids" (2024) is seven years. The gap between GridFM's agenda paper and
working open implementations is under two. Tutorial 10 sits exactly here.

---

## A second timeline: where the data lives

*Tutorial 11 (optional/advanced).* Model generality and data decentralization
are independent axes, and they have separate histories.

| Year | Milestone |
|---|---|
| **2016** | **FedAvg** — McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*. The algorithm that named the field. |
| **2016** | **DP-SGD** — Abadi et al. Clipping, calibrated noise and the moments accountant, still the template. |
| **2017** | **Secure aggregation** — Bonawitz et al. The server learns the sum and nothing else. |
| **2019** | **Deep Leakage from Gradients** — Zhu et al. reconstruct training samples from a shared gradient, ending "gradients are not data". |
| **2020** | **FedProx** and **SCAFFOLD** — two answers to client drift: penalise it, or correct it. |
| **2021** | **Kairouz et al.** — the field's reference survey; federated learning becomes a research area rather than an algorithm. |
| **2022** | **LoRA** (ICLR) — not federated, but the precondition for federating anything large. |
| **2023–24** | **Federated foundation models** — the two literatures meet. Surveys by Zhuang et al. (2023), Woisetschläger et al. (IJCAI 2024) and Ren et al. (2024) map a field that is roughly two years old. |
| **2024–25** | **Federated PEFT** — FFA-LoRA, FlexLoRA and successors confront the fact that averaging $A$ and $B$ separately is not averaging $BA$. |
| **2024–** | **Energy applications** — federated load forecasting and smart-grid surveys appear; no mature multi-operator deployment is public. |

Two observations worth carrying into Tutorial 11.

**Federated learning is older than the foundation-model era.** FedAvg predates
the Transformer by a year. The hard part was never the averaging; it was
heterogeneity, and then it was scale.

**The privacy attacks arrived after the architecture.** FedAvg is from 2016 and
*Deep Leakage from Gradients* from 2019 — three years in which "the data stays
local" was widely read as "the data is private". That gap is why this course
states the distinction every time it comes up rather than once.
