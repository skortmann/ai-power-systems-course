"""Message passing on power-system graphs — the backbone of the Mini-GridFM.

Why a graph model at all? A feed-forward network takes a fixed-width input, so
a model trained on a 30-bus network cannot even be *evaluated* on a 118-bus
one. Message passing has no such constraint: the parameters live in the message
and update functions, which are shared across every bus and every branch. The
same weights therefore apply to a grid the model has never seen — which is the
minimum requirement for anything calling itself a foundation model for grids.

Implemented with plain PyTorch (``index_add_``) rather than PyTorch Geometric:
the scatter-add is three lines, and seeing it removes the magic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = [
    "GraphBatch",
    "collate_graphs",
    "MessagePassingLayer",
    "GridEncoder",
    "MaskedGridModel",
    "NodeHead",
    "GraphHead",
    "global_mean_pool",
]


@dataclass
class GraphBatch:
    """Several graphs concatenated into one disconnected graph.

    ``node_features``  ``(N_total, F_node)``
    ``edge_index``     ``(2, E_total)``, already offset per graph
    ``edge_features``  ``(E_total, F_edge)``
    ``batch``          ``(N_total,)`` mapping each node to its graph
    ``n_graphs``       how many graphs were merged

    Merging graphs instead of padding them to a common size is what keeps the
    model independent of the largest grid in the dataset.
    """

    node_features: Tensor
    edge_index: Tensor
    edge_features: Tensor
    batch: Tensor
    n_graphs: int

    def to(self, device: str | torch.device) -> GraphBatch:
        return GraphBatch(
            self.node_features.to(device),
            self.edge_index.to(device),
            self.edge_features.to(device),
            self.batch.to(device),
            self.n_graphs,
        )

    @property
    def n_nodes(self) -> int:
        return int(self.node_features.shape[0])


def collate_graphs(graphs: list[dict[str, Tensor]]) -> GraphBatch:
    """Merge a list of ``{node_features, edge_index, edge_features}`` dicts."""
    nodes, edges, edge_attrs, batch_ids = [], [], [], []
    offset = 0
    for i, graph in enumerate(graphs):
        node_features = graph["node_features"]
        nodes.append(node_features)
        edges.append(graph["edge_index"] + offset)
        edge_attrs.append(graph["edge_features"])
        batch_ids.append(torch.full((node_features.shape[0],), i, dtype=torch.long))
        offset += node_features.shape[0]
    return GraphBatch(
        node_features=torch.cat(nodes, dim=0),
        edge_index=torch.cat(edges, dim=1),
        edge_features=torch.cat(edge_attrs, dim=0),
        batch=torch.cat(batch_ids, dim=0),
        n_graphs=len(graphs),
    )


def global_mean_pool(x: Tensor, batch: Tensor, n_graphs: int) -> Tensor:
    """Average node embeddings within each graph -> ``(n_graphs, d)``.

    Permutation-invariant by construction: relabelling the buses cannot change
    the graph-level embedding. That invariance is exactly what lets an
    "operating state" embedding mean the same thing across different grids.
    """
    totals = x.new_zeros(n_graphs, x.shape[1]).index_add_(0, batch, x)
    counts = x.new_zeros(n_graphs).index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    return totals / counts.clamp(min=1).unsqueeze(1)


class MessagePassingLayer(nn.Module):
    r"""One round of neighbourhood aggregation.

    .. math::
        m_{i \to j} = \phi([\, h_i,\; h_j,\; e_{ij} \,]), \qquad
        h_j' = h_j + \psi([\, h_j,\; \textstyle\sum_{i \in N(j)} m_{i \to j} \,])

    Sum aggregation is deliberate. Mean aggregation would make a bus with two
    incident lines indistinguishable from a bus with twenty carrying half the
    current each, and total injected power is a *sum* over branches — the
    physics is additive, so the aggregator should be too.

    The residual ``h_j +`` keeps deep stacks trainable and lets a layer refine
    rather than overwrite what the previous one found.
    """

    def __init__(self, d_model: int, edge_dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or 2 * d_model
        self.message = nn.Sequential(
            nn.Linear(2 * d_model + edge_dim, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )
        self.update = nn.Sequential(
            nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
        source, target = edge_index[0], edge_index[1]
        messages = self.message(
            torch.cat([x[source], x[target], edge_features], dim=-1)
        )
        aggregated = torch.zeros_like(x).index_add_(0, target, messages)
        return self.norm(x + self.update(torch.cat([x, aggregated], dim=-1)))


class GridEncoder(nn.Module):
    """Stack of message-passing layers producing one embedding per bus.

    The receptive field grows by one hop per layer, so ``n_layers`` controls how
    far electrical influence can propagate inside the model. Four layers reach
    four buses away — enough for a distribution feeder, deliberately not enough
    to see a whole transmission system, which is a real and teachable limit.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        d_model: int = 64,
        n_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(node_dim, d_model)
        self.layers = nn.ModuleList(
            MessagePassingLayer(d_model, edge_dim) for _ in range(n_layers)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: GraphBatch) -> Tensor:
        """Returns node embeddings of shape ``(N_total, d_model)``."""
        x = self.input_proj(batch.node_features)
        for layer in self.layers:
            x = self.dropout(layer(x, batch.edge_index, batch.edge_features))
        return x


class NodeHead(nn.Module):
    """Per-bus prediction head (masked-feature reconstruction, voltage, ...)."""

    def __init__(self, d_model: int, n_outputs: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, n_outputs)
        )

    def forward(self, node_embeddings: Tensor) -> Tensor:
        return self.net(node_embeddings)


class GraphHead(nn.Module):
    """Graph-level head on top of mean-pooled node embeddings."""

    def __init__(self, d_model: int, n_outputs: int = 1, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, n_outputs)
        )

    def forward(self, node_embeddings: Tensor, batch: Tensor, n_graphs: int) -> Tensor:
        return self.net(global_mean_pool(node_embeddings, batch, n_graphs))


class MaskedGridModel(nn.Module):
    """Encoder + reconstruction head: the Mini-GridFM's pretraining model.

    The self-supervised task is BERT's, transplanted onto a power grid: hide
    some bus features, reconstruct them from the rest of the network. Doing
    that well requires an internal model of how injections, voltages and
    topology constrain one another — which is precisely the reusable knowledge
    we want the encoder to keep.

    After pretraining the head is thrown away and ``encoder`` is the artefact.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        d_model: int = 64,
        n_layers: int = 4,
        n_state: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        #: How many leading node features are *state* (maskable and
        #: reconstructed). The remaining features are static equipment
        #: descriptors that stay visible, because an operator always knows the
        #: bus type and voltage level even when a measurement drops out.
        self.n_state = node_dim if n_state is None else n_state
        self.encoder = GridEncoder(node_dim, edge_dim, d_model, n_layers, dropout)
        self.head = NodeHead(d_model, self.n_state)
        #: Learned replacement for hidden features, like BERT's [MASK] token.
        self.mask_token = nn.Parameter(torch.zeros(self.n_state))

    def apply_mask(self, node_features: Tensor, mask: Tensor) -> Tensor:
        """Replace the state block of masked nodes with the learned mask token."""
        masked = node_features.clone()
        state = masked[:, : self.n_state]
        masked[:, : self.n_state] = torch.where(
            mask.unsqueeze(-1), self.mask_token.expand_as(state), state
        )
        return masked

    def forward(self, batch: GraphBatch, mask: Tensor | None = None) -> Tensor:
        """``mask``: ``(N_total,)`` boolean, ``True`` where the bus is hidden.

        Returns the reconstructed state features for **every** node; the loss
        selects the masked ones.
        """
        if mask is not None:
            batch = GraphBatch(
                self.apply_mask(batch.node_features, mask),
                batch.edge_index,
                batch.edge_features,
                batch.batch,
                batch.n_graphs,
            )
        return self.head(self.encoder(batch))

    @torch.no_grad()
    def embed(self, batch: GraphBatch) -> Tensor:
        self.eval()
        return self.encoder(batch)
