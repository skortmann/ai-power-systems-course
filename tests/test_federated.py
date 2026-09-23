"""Federated learning: aggregation, accounting, and the privacy primitives.

The load-bearing tests here are the ones that catch a federation which *looks*
like it works. Unweighted averaging, in-place aggregation that mutates a
client's tensors, and a LoRA payload that quietly includes the frozen backbone
all produce plausible numbers and no error, so each has an explicit test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from ai_power_course.federated import (
    ClientData,
    clip_update,
    communication_table,
    count_parameters,
    federated_average,
    gaussian_noise,
    local_train,
    parameter_delta,
    run_federation,
    state_dict_bytes,
    trainable_state_dict,
)


def _state(w, b):
    return {"w": torch.tensor(w), "b": torch.tensor(b)}


# --- aggregation -------------------------------------------------------------


def test_federated_average_weights_by_dataset_size():
    """The defining property. Unweighted averaging is a different algorithm."""
    a = _state([1.0, 2.0], [0.0])
    b = _state([3.0, 4.0], [2.0])
    out = federated_average([a, b], [1, 3])

    assert out["w"] == pytest.approx([2.5, 3.5])
    assert out["b"] == pytest.approx([1.5])
    # The unweighted answer, which must NOT be what we got.
    assert out["w"] != pytest.approx([2.0, 3.0])


def test_federated_average_does_not_mutate_its_inputs():
    """A server that edits a client's tensors converges to nonsense slowly."""
    a = _state([1.0, 2.0], [0.0])
    b = _state([3.0, 4.0], [2.0])
    before = a["w"].clone()
    federated_average([a, b], [1, 1])
    assert torch.equal(a["w"], before)


def test_federated_average_preserves_keys_shapes_and_dtype():
    a = {"w": torch.zeros(3, 4), "b": torch.zeros(4)}
    b = {"w": torch.ones(3, 4), "b": torch.ones(4)}
    out = federated_average([a, b], [1, 1])
    assert set(out) == {"w", "b"}
    assert out["w"].shape == (3, 4)
    assert out["w"].dtype == a["w"].dtype


def test_federated_average_of_identical_clients_is_the_input():
    a = _state([1.5, -2.5], [0.25])
    out = federated_average([a, a, a], [7, 7, 7])
    assert out["w"] == pytest.approx(a["w"].tolist())


def test_federated_average_rejects_mismatched_keys():
    a = {"w": torch.zeros(2)}
    b = {"v": torch.zeros(2)}
    with pytest.raises(ValueError, match="different parameter keys"):
        federated_average([a, b], [1, 1])


def test_federated_average_rejects_empty_and_zero_sizes():
    with pytest.raises(ValueError):
        federated_average([], [])
    with pytest.raises(ValueError):
        federated_average([_state([1.0], [0.0])], [0])


def test_parameter_delta_is_the_difference():
    new = _state([3.0], [1.0])
    old = _state([1.0], [0.5])
    delta = parameter_delta(new, old)
    assert delta["w"] == pytest.approx([2.0])
    assert delta["b"] == pytest.approx([0.5])


# --- communication accounting ------------------------------------------------


def test_state_dict_bytes_uses_the_real_dtype():
    """Hard-coding 4 bytes/value is right for float32 and wrong the moment
    anyone quantises -- which is a standard communication-reduction trick."""
    assert state_dict_bytes({"x": torch.zeros(10, dtype=torch.float32)}) == 40
    assert state_dict_bytes({"x": torch.zeros(10, dtype=torch.float16)}) == 20
    assert state_dict_bytes({}) == 0


def test_communication_table_counts_both_directions():
    payload = {"x": torch.zeros(100, dtype=torch.float32)}   # 400 bytes
    rows = communication_table({"m": payload}, n_rounds=10, n_clients=4)
    row = rows[0]
    assert row["parameters"] == 100
    assert row["bytes_per_message"] == 400
    # down to 4 clients and back up again
    assert row["MB_per_round"] == pytest.approx(400 * 4 * 2 / 1e6)
    assert row["MB_total"] == pytest.approx(400 * 4 * 2 * 10 / 1e6)


def test_trainable_state_dict_excludes_frozen_parameters():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    for param in model[0].parameters():
        param.requires_grad = False

    payload = trainable_state_dict(model)
    assert all(not k.startswith("0.") for k in payload), (
        f"frozen layer leaked into the payload: {sorted(payload)}"
    )
    assert any(k.startswith("1.") for k in payload)

    trainable, total = count_parameters(model)
    assert trainable < total


# --- the privacy primitives --------------------------------------------------


def test_clip_update_bounds_the_global_norm():
    update = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([12.0])}
    before = float(sum((t**2).sum() for t in update.values()) ** 0.5)
    assert before == pytest.approx(13.0)

    clipped = clip_update(update, max_norm=1.0)
    after = float(sum((t**2).sum() for t in clipped.values()) ** 0.5)
    assert after == pytest.approx(1.0)


def test_clip_update_leaves_a_small_update_alone():
    update = {"a": torch.tensor([0.1, 0.1])}
    clipped = clip_update(update, max_norm=10.0)
    assert clipped["a"] == pytest.approx(update["a"].tolist())


def test_clip_update_does_not_mutate_and_rejects_bad_norms():
    update = {"a": torch.tensor([3.0, 4.0])}
    before = update["a"].clone()
    clip_update(update, max_norm=1.0)
    assert torch.equal(update["a"], before)
    with pytest.raises(ValueError):
        clip_update(update, max_norm=0.0)


def test_gaussian_noise_is_zero_mean_and_reproducible():
    update = {"a": torch.zeros(20_000)}
    noised = gaussian_noise(
        update, std=0.5, generator=torch.Generator().manual_seed(0)
    )
    assert float(noised["a"].mean()) == pytest.approx(0.0, abs=0.02)
    assert float(noised["a"].std()) == pytest.approx(0.5, rel=0.05)

    again = gaussian_noise(
        update, std=0.5, generator=torch.Generator().manual_seed(0)
    )
    assert torch.equal(noised["a"], again["a"])


def test_zero_noise_is_a_no_op():
    update = {"a": torch.tensor([1.0, 2.0])}
    assert gaussian_noise(update, std=0.0)["a"] == pytest.approx([1.0, 2.0])


# --- the loop ----------------------------------------------------------------


def _toy_clients(n_clients: int = 3, n: int = 64, seed: int = 0):
    """Clients on a shared linear relationship, so a federation can succeed."""
    clients = []
    for index in range(n_clients):
        rng = np.random.default_rng(seed + index)
        x = torch.tensor(rng.normal(size=(n, 3)), dtype=torch.float32)
        y = (x @ torch.tensor([1.0, -2.0, 0.5])).unsqueeze(1)
        clients.append(
            ClientData(
                name=f"C{index}",
                x_train=x[: n // 2], y_train=y[: n // 2],
                x_valid=x[n // 2 :], y_valid=y[n // 2 :],
            )
        )
    return clients


def _make_model():
    return nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))


def test_run_federation_improves_and_records_per_client_results():
    torch.manual_seed(0)
    clients = _toy_clients()
    _, history = run_federation(
        _make_model, clients, n_rounds=8, local_epochs=5, lr=0.05, seed=0
    )

    assert len(history.rounds) == 8
    assert history.global_mae[-1] < history.global_mae[0], (
        f"federation did not improve: {history.global_mae[0]:.4f} -> "
        f"{history.global_mae[-1]:.4f}"
    )
    assert set(history.per_client_mae[-1]) == {c.name for c in clients}
    assert np.isfinite(history.worst_client())
    assert history.worst_client() >= np.mean(list(history.per_client_mae[-1].values()))


def test_run_federation_is_reproducible():
    clients = _toy_clients()
    torch.manual_seed(0)
    _, first = run_federation(_make_model, clients, n_rounds=3, local_epochs=2, seed=7)
    torch.manual_seed(0)
    _, second = run_federation(_make_model, clients, n_rounds=3, local_epochs=2, seed=7)
    assert first.global_mae == pytest.approx(second.global_mae)


def test_fedprox_reduces_client_drift():
    """mu > 0 must move drift in the direction the proximal term is designed to.

    Deliberately tests the direction rather than an accuracy improvement:
    FedProx constrains drift, and whether that buys accuracy depends on the
    federation.
    """
    clients = _toy_clients()
    torch.manual_seed(0)
    _, plain = run_federation(
        _make_model, clients, n_rounds=5, local_epochs=20, lr=0.05, mu=0.0, seed=0
    )
    torch.manual_seed(0)
    _, prox = run_federation(
        _make_model, clients, n_rounds=5, local_epochs=20, lr=0.05, mu=1.0, seed=0
    )
    assert prox.drift[-1] < plain.drift[-1], (
        f"FedProx drift {prox.drift[-1]:.5f} should be below FedAvg's "
        f"{plain.drift[-1]:.5f}"
    )


def test_participation_below_one_samples_a_subset():
    clients = _toy_clients(n_clients=4)
    torch.manual_seed(0)
    _, history = run_federation(
        _make_model, clients, n_rounds=4, local_epochs=2, participation=0.5, seed=1
    )
    # Every client is still EVALUATED each round even when not selected.
    assert set(history.per_client_mae[-1]) == {c.name for c in clients}
    assert len(history.local_losses[-1]) == 2


def test_noise_degrades_accuracy():
    """The privacy-utility trade-off, as a direction rather than a magnitude."""
    clients = _toy_clients()
    torch.manual_seed(0)
    _, clean = run_federation(
        _make_model, clients, n_rounds=6, local_epochs=5, lr=0.05, seed=0
    )
    torch.manual_seed(0)
    _, noisy = run_federation(
        _make_model, clients, n_rounds=6, local_epochs=5, lr=0.05,
        clip_norm=0.5, noise_std=0.5, seed=0,
    )
    assert noisy.global_mae[-1] > clean.global_mae[-1], (
        "heavy noise should cost accuracy; if it does not, the noise is not "
        "reaching the aggregated update"
    )


def test_local_train_with_mu_requires_a_global_state():
    model = _make_model()
    x = torch.zeros(4, 3)
    y = torch.zeros(4, 1)
    with pytest.raises(ValueError, match="FedProx"):
        local_train(model, x, y, mu=0.1)
