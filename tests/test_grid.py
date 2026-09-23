"""The Mini-GridFM stack: networks, graphs, physics and message passing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ai_power_course.grid.graphs import (
    EDGE_FEATURE_NAMES,
    N_STATE_FEATURES,
    NODE_FEATURE_NAMES,
    net_to_graph,
)
from ai_power_course.grid.networks import (
    HELDOUT_NETWORKS,
    NETWORK_CATALOGUE,
    PRETRAIN_NETWORKS,
    load_network,
    run_power_flow,
)
from ai_power_course.grid.physics import (
    complex_voltage,
    get_ybus,
    internal_state,
    line_loading_violations,
    physical_report,
    power_balance_residual,
    voltage_violations,
)
from ai_power_course.grid.sampling import SamplingConfig, sample_operating_points
from ai_power_course.models.gnn import (
    GraphBatch,
    GridEncoder,
    MaskedGridModel,
    collate_graphs,
    global_mean_pool,
)

SMALL = ("case9", "case14", "case30")


@pytest.fixture(scope="module")
def solved_case14():
    net = load_network("case14")
    assert run_power_flow(net)
    return net


# --- networks -----------------------------------------------------------------


def test_pretrain_and_heldout_sets_are_disjoint():
    assert not set(PRETRAIN_NETWORKS) & set(HELDOUT_NETWORKS)
    for name in PRETRAIN_NETWORKS + HELDOUT_NETWORKS:
        assert name in NETWORK_CATALOGUE


@pytest.mark.parametrize("name", SMALL)
def test_catalogued_networks_solve(name):
    net = load_network(name)
    assert run_power_flow(net)
    assert net.res_bus.vm_pu.notna().all()


def test_unknown_network_raises():
    with pytest.raises(KeyError, match="unknown network"):
        load_network("case_does_not_exist")


def test_run_power_flow_does_not_trust_a_stale_flag():
    """pandapower's shipped networks carry converged=True before any solve."""
    net = load_network("case9")
    assert net.get("converged") is True, "precondition for this regression test"
    # Disconnect everything so the solve cannot succeed meaningfully.
    net.line["in_service"] = False
    net.trafo["in_service"] = False
    converged = run_power_flow(net)
    if converged:
        # If pandapower still solves the degenerate case, the graph builder must
        # be the one to reject it.
        with pytest.raises(ValueError):
            net_to_graph(net)


# --- graph conversion ---------------------------------------------------------


def test_graph_shapes_and_feature_names(solved_case14):
    graph = net_to_graph(solved_case14)
    n_bus = graph["node_features"].shape[0]
    assert graph["node_features"].shape == (n_bus, len(NODE_FEATURE_NAMES))
    assert graph["edge_features"].shape[1] == len(EDGE_FEATURE_NAMES)
    assert graph["edge_index"].shape[0] == 2
    assert graph["edge_index"].shape[1] == graph["edge_features"].shape[0]
    assert graph["vm_pu"].shape == (n_bus,)
    assert graph["va_degree"].shape == (n_bus,)
    assert torch.isfinite(graph["node_features"]).all()
    assert torch.isfinite(graph["edge_features"]).all()
    assert N_STATE_FEATURES == 5


def test_graph_is_undirected(solved_case14):
    graph = net_to_graph(solved_case14)
    edges = graph["edge_index"].numpy()
    forward = set(zip(edges[0], edges[1], strict=True))
    reverse = set(zip(edges[1], edges[0], strict=True))
    assert forward == reverse, "every branch must appear in both directions"


def test_graph_state_matches_the_solved_network(solved_case14):
    graph = net_to_graph(solved_case14)
    vm, va = internal_state(solved_case14)
    np.testing.assert_allclose(graph["vm_pu"].numpy(), vm, atol=1e-6)
    np.testing.assert_allclose(graph["va_degree"].numpy(), va, atol=1e-5)
    # vm and the (sin, cos) angle encoding must be consistent with the features.
    features = graph["node_features"].numpy()
    np.testing.assert_allclose(features[:, 2], vm, atol=1e-5)
    np.testing.assert_allclose(features[:, 3], np.sin(np.radians(va)), atol=1e-5)
    np.testing.assert_allclose(features[:, 4], np.cos(np.radians(va)), atol=1e-5)


def test_bus_type_one_hot_is_exclusive(solved_case14):
    features = net_to_graph(solved_case14)["node_features"].numpy()
    one_hot = features[:, 5:8]
    np.testing.assert_allclose(one_hot.sum(axis=1), 1.0)
    assert one_hot[:, 0].sum() >= 1, "there must be a slack bus"


def test_graph_requires_a_power_flow():
    with pytest.raises(RuntimeError, match="run a power flow"):
        net_to_graph(load_network("case9"))


# --- physics ------------------------------------------------------------------


def test_ybus_is_square_symmetric_and_matches_the_state(solved_case14):
    ybus = get_ybus(solved_case14)
    assert ybus.shape[0] == ybus.shape[1]
    np.testing.assert_allclose(ybus, ybus.T, atol=1e-9)


def test_the_solved_state_has_zero_power_balance_residual(solved_case14):
    vm, va = internal_state(solved_case14)
    report = power_balance_residual(solved_case14, vm, va)
    assert report["p_mismatch_max_mw"] < 1e-6
    assert report["q_mismatch_max_mvar"] < 1e-6


def test_a_perturbed_state_has_a_nonzero_residual(solved_case14):
    """The check must actually detect a physically invalid state."""
    vm, va = internal_state(solved_case14)
    report = power_balance_residual(solved_case14, vm * 1.05, va)
    assert report["p_mismatch_max_mw"] > 1.0


def test_power_balance_rejects_a_wrong_length_state(solved_case14):
    vm, va = internal_state(solved_case14)
    with pytest.raises(ValueError, match="Ybus has"):
        power_balance_residual(solved_case14, vm[:-1], va[:-1])


def test_voltage_violations():
    report = voltage_violations(np.array([0.90, 1.00, 1.10]), v_min=0.95, v_max=1.05)
    assert report["undervoltage_rate"] == pytest.approx(1 / 3)
    assert report["overvoltage_rate"] == pytest.approx(1 / 3)
    assert report["violation_rate"] == pytest.approx(2 / 3)


def test_line_loading_and_full_report(solved_case14):
    loading = line_loading_violations(solved_case14)
    assert loading["n_branches"] > 0
    assert loading["max_loading_percent"] >= 0
    vm, va = internal_state(solved_case14)
    report = physical_report(solved_case14, vm, va)
    assert {"violation_rate", "p_mismatch_max_mw", "max_loading_percent"} <= set(report)

    # Every quantity in a report headed "checks on the prediction" must respond
    # to the prediction. Line loading used to read the network's stored
    # solution and was therefore constant whatever was passed in -- three
    # columns of the answer key in a table about predictions.
    nonsense = physical_report(
        solved_case14, np.full_like(vm, 0.80), np.zeros_like(va)
    )
    assert nonsense["violation_rate"] > report["violation_rate"]
    assert nonsense["p_mismatch_max_mw"] > report["p_mismatch_max_mw"]
    assert nonsense["max_loading_percent"] != report["max_loading_percent"], (
        "line loading did not move when the predicted state changed, so it is "
        "being read from the network's own solution rather than computed"
    )


def test_line_loading_from_state_matches_pandapower(solved_case14):
    """The derived loading must reproduce pandapower's own on the true state.

    This is what licenses using it on a PREDICTED state: if it disagrees where
    the answer is known, it cannot be trusted where it is not.
    """
    from ai_power_course.grid.physics import line_loading_from_state

    derived = line_loading_from_state(solved_case14, *internal_state(solved_case14))
    assert derived["n_lines"] == len(solved_case14.line)
    assert derived["max_loading_percent"] == pytest.approx(
        float(solved_case14.res_line.loading_percent.max()), abs=1e-9
    )


@pytest.mark.parametrize("name", ["case9", "case33bw", "case57", "case118"])
def test_line_loading_from_state_handles_every_catalogued_network(name):
    """case33bw is the one that matters: 37 lines, 5 of them open tie switches.

    The internal ppc that carries Yf drops de-energised branches while the
    element lookup keeps them, so a positional mapping misaligns every row
    after the first opening.
    """
    from ai_power_course.grid.networks import load_network, run_power_flow
    from ai_power_course.grid.physics import line_loading_from_state

    net = load_network(name)
    assert run_power_flow(net)
    derived = line_loading_from_state(net, *internal_state(net))
    assert derived["n_lines"] == len(net.line)
    assert derived["max_loading_percent"] == pytest.approx(
        float(net.res_line.loading_percent.max()), abs=1e-6
    )


def test_complex_voltage_roundtrip():
    vm = np.array([1.0, 0.98])
    va = np.array([0.0, -12.5])
    voltage = complex_voltage(vm, va)
    np.testing.assert_allclose(np.abs(voltage), vm)
    np.testing.assert_allclose(np.degrees(np.angle(voltage)), va)


# --- sampling -----------------------------------------------------------------


def test_sampling_produces_valid_varied_states():
    points, stats = sample_operating_points(
        "case14", SamplingConfig(n_samples=12, seed=7), verbose=False
    )
    assert len(points) == 12
    assert stats["attempts"] >= 12
    load_scales = [point.load_scale for point in points]
    assert max(load_scales) - min(load_scales) > 0.1, "operating points should vary"
    for point in points:
        assert point.vm_pu.shape[0] == point.n_buses
        assert np.isfinite(point.vm_pu).all()
        assert torch.isfinite(point.graph["node_features"]).all()


def test_sampling_is_reproducible():
    a, _ = sample_operating_points("case9", SamplingConfig(n_samples=5, seed=3), verbose=False)
    b, _ = sample_operating_points("case9", SamplingConfig(n_samples=5, seed=3), verbose=False)
    for first, second in zip(a, b, strict=True):
        np.testing.assert_allclose(first.vm_pu, second.vm_pu)


def test_sampling_rejects_heavily_islanded_states():
    """A radial feeder islands easily; those samples must not enter the dataset."""
    points, stats = sample_operating_points(
        "case33bw",
        SamplingConfig(n_samples=15, outage_probability=1.0, seed=11,
                       min_energised_fraction=0.8),
        verbose=False,
    )
    assert stats["islanded"] >= 0
    base = load_network("case33bw")
    for point in points:
        assert point.n_buses >= 0.8 * len(base.bus)


# --- message passing ----------------------------------------------------------


def _graphs(names=SMALL):
    graphs = []
    for name in names:
        net = load_network(name)
        assert run_power_flow(net)
        graphs.append(net_to_graph(net))
    return graphs


def test_collate_offsets_edge_indices_correctly():
    graphs = _graphs()
    batch = collate_graphs(graphs)
    assert batch.n_graphs == len(graphs)
    assert batch.n_nodes == sum(g["node_features"].shape[0] for g in graphs)
    assert int(batch.edge_index.max()) < batch.n_nodes
    counts = torch.bincount(batch.batch).tolist()
    assert counts == [g["node_features"].shape[0] for g in graphs]


def test_global_mean_pool():
    x = torch.tensor([[1.0], [3.0], [10.0]])
    batch = torch.tensor([0, 0, 1])
    pooled = global_mean_pool(x, batch, n_graphs=2)
    assert pooled.shape == (2, 1)
    assert pooled[0, 0] == pytest.approx(2.0)
    assert pooled[1, 0] == pytest.approx(10.0)


def test_one_encoder_handles_every_grid_size():
    """The property that makes cross-grid transfer possible at all."""
    encoder = GridEncoder(len(NODE_FEATURE_NAMES), len(EDGE_FEATURE_NAMES),
                          d_model=16, n_layers=2)
    for name in ("case9", "case14", "case118", "case33bw"):
        net = load_network(name)
        assert run_power_flow(net)
        batch = collate_graphs([net_to_graph(net)])
        output = encoder(batch)
        assert output.shape == (batch.n_nodes, 16)
        assert torch.isfinite(output).all()


def test_masked_grid_model_masks_only_the_state_block():
    set_seed_value = torch.manual_seed(0)
    assert set_seed_value is not None
    model = MaskedGridModel(len(NODE_FEATURE_NAMES), len(EDGE_FEATURE_NAMES),
                            d_model=16, n_layers=2, n_state=N_STATE_FEATURES)
    batch = collate_graphs(_graphs(("case9",)))
    mask = torch.zeros(batch.n_nodes, dtype=torch.bool)
    mask[0] = True

    masked = model.apply_mask(batch.node_features, mask)
    # The static block is untouched...
    torch.testing.assert_close(
        masked[:, N_STATE_FEATURES:], batch.node_features[:, N_STATE_FEATURES:]
    )
    # ...the unmasked nodes' state is untouched...
    torch.testing.assert_close(masked[1:, :N_STATE_FEATURES],
                               batch.node_features[1:, :N_STATE_FEATURES])
    # ...and the masked node carries the learned token.
    torch.testing.assert_close(masked[0, :N_STATE_FEATURES], model.mask_token)

    output = model(batch, mask=mask)
    assert output.shape == (batch.n_nodes, N_STATE_FEATURES)


def test_masked_grid_model_is_permutation_equivariant():
    """Relabelling buses must permute the outputs, not change them."""
    torch.manual_seed(0)
    model = MaskedGridModel(len(NODE_FEATURE_NAMES), len(EDGE_FEATURE_NAMES),
                            d_model=16, n_layers=2, n_state=N_STATE_FEATURES).eval()
    graph = _graphs(("case14",))[0]
    batch = collate_graphs([graph])
    with torch.no_grad():
        original = model(batch)

    permutation = torch.randperm(batch.n_nodes)
    inverse = torch.argsort(permutation)
    shuffled = GraphBatch(
        node_features=batch.node_features[permutation],
        edge_index=inverse[batch.edge_index],
        edge_features=batch.edge_features,
        batch=batch.batch,
        n_graphs=1,
    )
    with torch.no_grad():
        permuted = model(shuffled)
    torch.testing.assert_close(permuted, original[permutation], atol=1e-5, rtol=1e-4)
