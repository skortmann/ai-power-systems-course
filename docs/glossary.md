# Glossary

Definitions as this course uses them. Where a term is contested or commonly
misused, that is said rather than glossed over.

Terms are grouped by where they first appear. The tutorial number in brackets
points at where the idea is built rather than merely mentioned.

---

## Learning paradigms

**Artificial intelligence (AI)** — the broad field of building systems that
perform tasks associated with intelligence. In current usage outside research
it usually means machine learning, and increasingly it means deep learning
specifically. Vague by construction; prefer a narrower term when you have one.

**Machine learning (ML)** — fitting a function to data rather than specifying it
by hand. [01]

**Deep learning** — machine learning with multi-layer neural networks, where the
intermediate representations are learned rather than engineered. [02]

**Supervised learning** — learning a mapping from inputs to *given* labels. [01]

**Unsupervised learning** — finding structure in data with no labels at all
(clustering, dimensionality reduction).

**Self-supervised learning** — manufacturing a supervised problem out of the
data's own structure: hide part of the input and predict it, or make two views
of the same thing agree. No human annotation, but a real prediction target.
This is how essentially every foundation model is pretrained. [04]

**Transfer learning** — reusing a model trained on one task or dataset as the
starting point for another. [04, 08]

**In-context learning** — a model performing a task specified entirely in its
prompt, with no weight update. Nothing persists after the context window
closes. Strongly scale-dependent. [08]

---

## The mechanics

**Feature** — an input variable. In classical ML, chosen by a human; in deep
learning, increasingly computed by the model itself. [01, 02]

**Label** — the target the model is trained to predict. [01]

**Parameter** — a number learned during training (a weight or a bias). Not the
same as a hyper-parameter, which you choose. Parameter counts are *not*
comparable across model families: a decision-tree node and a network weight are
different things. [02]

**Tensor** — a multi-dimensional array, plus (in PyTorch) a record of the
operations that produced it, which is what makes automatic differentiation
possible. [02]

**Gradient** — the vector of partial derivatives of the loss with respect to the
parameters. It points in the direction of steepest increase, so training steps
the other way. [02]

**Backpropagation** — computing those gradients efficiently by applying the
chain rule backwards through the network, reusing intermediate results. It is
bookkeeping, not a learning algorithm; the learning algorithm is gradient
descent. [02]

**Epoch** — one complete pass over the training set. [02]

**Batch (mini-batch)** — the subset of examples used for one parameter update.
Smaller batches mean noisier gradients and more updates per epoch. [02]

**Learning rate** — the step size of the optimiser. The hyper-parameter most
likely to decide whether training works at all. [02]

**Overfitting** — fitting the training data better than the underlying pattern,
so performance on new data degrades. Diagnosed by the gap between training and
validation curves, not by the training curve alone. [01]

**Data leakage** — information reaching the model that would not be available at
prediction time. It produces excellent results and no error message. In time
series the classic cause is a random rather than chronological split. [01]

---

## Representations

**Representation** — the internal vector a model computes for an input, before
its final layer. [04]

**Latent space** — the space those vectors live in. [04]

**Embedding** — one input's vector in that space. The useful property is that
distance in the space corresponds to similarity in a way you care about. [04, 08]

**Autoencoder** — a model trained to reconstruct its input through a bottleneck.
Its weakness is that a wide enough bottleneck lets it learn the identity
function. [04]

**Masked modelling** — hiding part of the input and predicting it from the rest,
scoring only the hidden part. BERT for text, MAE for images, and the
pretraining objective of most time-series and grid foundation models. [04, 10]

**Contrastive learning** — pulling two views of the same thing together in
representation space and pushing different things apart. The *choice of
augmentation* is the scientific content of the method, not a detail. [04]

**Linear probe** — a single linear layer fitted on frozen features, used to ask
how much a representation has disentangled. If a linear probe gets close to full
fine-tuning, the knowledge is in the representation, not the head. [04, 10]

---

## Attention and Transformers

**Attention** — a soft, differentiable dictionary lookup: score a query against
keys, normalise with a softmax, return a weighted average of values. [05]

**Query, key, value** — respectively "what am I looking for", "what does this
position offer", and "what this position returns if selected". [05]

**Self-attention** — attention where queries, keys and values all come from the
same sequence. [05]

**Causal (masked) attention** — self-attention where position $i$ may only
attend to $j \le i$. One line of code, and the entire difference between a model
that fills in blanks and one that predicts the future. [05]

**Positional encoding** — position information added to the representation,
necessary because attention is permutation-equivariant and would otherwise be
unable to tell 09:00 from 21:00. On a *graph* that same permutation equivariance
is a feature, which is why grid models treat position differently. [05, 10]

**Multi-head attention** — several attention operations in parallel over
subspaces of the representation, so the model can attend to "the last few hours"
and "the same hour yesterday" at once. [05]

**Transformer** — a stack of blocks, each containing multi-head attention, a
feed-forward network, residual connections and layer normalisation. **Not a
synonym for language model**: Transformers operate on any tokenisable modality.
[06]

**Encoder** — a Transformer with bidirectional attention. Good at understanding,
embedding and regression. BERT, DINOv2, Chronos-2, GridFM. [06]

**Decoder** — a Transformer with causal attention. Good at generation. GPT,
Llama, WindFM. [06]

**Encoder-decoder** — an encoder feeding a decoder through cross-attention.
Sequence-to-sequence tasks: T5, the original Transformer, Chronos-1. [06]

**Token** — whatever gets projected into the model's dimension. A subword in
text, a 16x16 patch in an image, one time step or patch of steps in a series,
one bus in a grid. [06, 07, 10]

**Vocabulary** — the finite set of tokens a text model can represent. Character
level is small and simple; subword (BPE, SentencePiece) is large and gives ~4x
shorter sequences, which matters because attention is quadratic in length. [07]

**Context window** — the maximum sequence length a model can attend over. A hard
architectural limit, and the reason long-context methods exist. [07]

**Teacher forcing** — training every position to predict its successor
simultaneously, using the true previous tokens as input. Makes the supervision
free: a length-$T$ sequence gives $T$ training signals. [07]

**Perplexity** — the exponential of the cross-entropy, read as "how many options
is the model effectively choosing between at each step". [07]

**Temperature, top-k, top-p** — controls that reshape the output distribution
before sampling. They change the *output*, never the model. [07]

---

## Pretraining and adaptation

**Pretraining** — the expensive, once-only training of a model on broad data,
usually with a self-supervised objective. [04, 09]

**Fine-tuning** — continuing training on a specific task, updating all weights.
[08]

**Frozen backbone** — using a pretrained model's weights unchanged and training
only a small head on top. [04, 08]

**PEFT (parameter-efficient fine-tuning)** — the family of methods that adapt a
model by training a small fraction of its parameters: adapters, prefix tuning,
LoRA. [08]

**LoRA (low-rank adaptation)** — freezing $W_0$ and learning a low-rank update
$\Delta W = BA$ with $r \ll d$. Typically well under 1% of the parameters, and at
inference $BA$ folds into $W_0$ so there is no latency cost. [08]

**Quantization** — storing weights at lower precision (8-bit, 4-bit) to cut
memory roughly four- or eightfold, at a small accuracy cost. [08]

**Zero-shot** — applying a model to a task with no task-specific examples. Note
that it means "no examples *of this task were given to it now*", **not** "the
model has never seen anything similar" — pretraining-distribution overlap and
benchmark contamination are real and often unmeasurable. [09]

**Few-shot** — a handful of examples, usually supplied in the prompt rather than
by training. [08]

**Instruction tuning** — supervised fine-tuning on (instruction, response) pairs.
This is what turns a text *completer* into a *responder*. [07]

**RLHF / preference optimisation** — further tuning against human preference
data, by reinforcement learning (RLHF) or directly (DPO). [07]

**RAG (retrieval-augmented generation)** — retrieving relevant documents and
putting them in the prompt. **A system architecture around a model, not a
training method and not a foundation model.** Not one weight changes. It buys
auditable sources, not correctness. [08]

---

## Foundation models

**Foundation model** — a model pretrained on broad data at scale, designed to be
adapted to many downstream tasks (after Bommasani et al., 2021). The load-bearing
words are **broad** and **adapted to many**. [09]

The distinctions this course insists on:

| term | means | example that is *only* this |
|---|---|---|
| **pretrained model** | trained before you got it | an ImageNet ResNet |
| **reusable backbone** | its representation transfers | a ResNet feature extractor |
| **foundation model** | broad pretraining + broad adaptability | DINOv2, Chronos-2 |
| **large language model** | a large model over text | a 70B text model |
| **generative model** | samples from a data distribution | a diffusion image model |

Every foundation model is pretrained; not every pretrained model is a foundation
model. Most current LLMs are foundation models; most foundation models are not
LLMs. DINOv2 generates nothing and is a foundation model; a 70B model
fine-tuned to do one thing is large and is not.

**Large language model (LLM)** — a large model trained on text with a
next-token objective. One column of the foundation-model table, not the table.

**Multimodal model** — one model consuming or producing more than one modality
(CLIP: image + text; RT-2: vision + language + action). [09]

**Time-series foundation model** — pretrained on many time series from many
domains, forecasting new ones zero-shot. Chronos-2, TimesFM, Moirai, MOMENT,
TinyTimeMixers. [09]

**Scaling laws** — empirical power laws relating loss to parameters, data and
compute (Kaplan et al., 2020; corrected by Hoffmann et al., 2022). They describe
*loss*, not capability, and scaling a model on a narrow corpus does not make it
foundational. [07]

**Mixture of experts (MoE)** — routing each token through a subset of the
parameters, so total capacity grows faster than compute per token.

**Benchmark contamination** — evaluation data having appeared in pretraining.
The reason "zero-shot" results on public datasets should be read sceptically,
and the reason this course evaluates on a dataset that provably cannot have been
pretrained on. [09]

---

## Power systems and grid AI

**Per-unit system** — expressing quantities as fractions of a chosen base, which
removes the voltage level from the numbers and makes a 380 kV line and a 20 kV
feeder comparable. Essential preprocessing for any model meant to transfer
between grids. [10]

**Power flow** — solving $S = V \odot (YV)^*$ for the complex bus voltages given
injections and the admittance matrix. Non-linear, solved iteratively, and not
guaranteed to have a solution. [10]

**Residual load** — demand minus non-dispatchable generation. More volatile than
demand, and the driver of negative prices. [01]

**N-1 criterion** — the system must stay within limits after the loss of any
single component. Assessed by running a power flow per contingency, which is why
fast surrogates are an active research area. [10]

**Persistence / seasonal naive** — the trivial forecasts every model must beat.
On hourly data at a 24-hour horizon they are the *same* forecast, which is worth
knowing before you report both. [01]

**Physical plausibility** — whether a prediction could occur in a real system:
non-negative generation, zero PV at night, voltages in band, power balance
satisfied, ramps within limits. A separate question from statistical accuracy,
and one no error metric answers. [01, 10]

**GNN (graph neural network)** — a model that operates on graphs by passing
messages along edges. Its parameters live in the message and update functions,
which are shared across every node and edge — so the *same weights* apply to a
graph of any size. That property is what makes transfer between grids possible
at all. [10]

**Message passing** — one round of a GNN: compute a message per edge, aggregate
at each node, update the node. Sum aggregation is usually right for power
systems because the physics is additive. [10]

**Graph transformer** — attention over graph nodes, with structure supplied by
masking or by structural encodings rather than by sequence position. [10]

**GridFM (grid foundation model)** — a foundation model for electric power
grids: pretrained across many networks and operating states, adaptable to power
flow, state estimation, contingency screening and more. As of 2026 an active
research area with open implementations (`gridfm-graphkit`, GridSFM) and no
mature operational deployment. [10]

**Topology-aware / permutation-equivariant** — a model whose output transforms
consistently when the buses are relabelled, and which does not assume a fixed
bus count. The minimum requirement for anything claiming to transfer across
grids. [10]

**Physics-informed loss** — adding a term penalising violation of a physical
law (here, the AC power-balance residual) to the training objective. [10]
