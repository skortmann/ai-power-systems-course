"""Forward passes, attention correctness, and the training loop."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from ai_power_course.config import set_seed
from ai_power_course.models.attention import (
    MultiHeadSelfAttention,
    causal_mask,
    scaled_dot_product_attention_numpy,
    sinusoidal_positional_encoding,
    softmax_numpy,
)
from ai_power_course.models.forecasters import (
    LSTMForecaster,
    MLPForecaster,
    SimpleRNNForecaster,
    TransformerBlock,
    TransformerForecaster,
)
from ai_power_course.models.representation import (
    LinearProbe,
    MaskedProfileAutoencoder,
    ProfileEncoder,
    block_mask,
    masked_reconstruction_loss,
    nt_xent_loss,
    random_mask,
)
from ai_power_course.models.tinygpt import CharTokenizer, TinyGPT, TinyGPTConfig, generate
from ai_power_course.models.training import (
    Standardizer,
    TrainConfig,
    count_parameters,
    make_loader,
    predict,
    train_model,
)

BATCH, CONTEXT, CHANNELS, HORIZON = 4, 32, 3, 8


# --- attention ----------------------------------------------------------------


def test_softmax_is_numerically_stable():
    huge = np.array([[1000.0, 1001.0, 999.0]])
    result = softmax_numpy(huge)
    assert np.isfinite(result).all()
    assert result.sum() == pytest.approx(1.0)


def test_our_attention_matches_pytorch():
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(6, 8)) for _ in range(3))
    ours, weights = scaled_dot_product_attention_numpy(q, k, v)
    reference = torch.nn.functional.scaled_dot_product_attention(
        *(torch.tensor(a).unsqueeze(0) for a in (q, k, v))
    ).squeeze(0).numpy()
    np.testing.assert_allclose(ours, reference, atol=1e-10)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_causal_mask_blocks_the_future_exactly():
    rng = np.random.default_rng(1)
    q, k, v = (rng.normal(size=(5, 4)) for _ in range(3))
    _, weights = scaled_dot_product_attention_numpy(q, k, v, mask=causal_mask(5).numpy())
    assert weights[np.triu_indices(5, k=1)].max() == 0.0
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_attention_is_permutation_equivariant():
    """The property that makes positional encoding necessary — and that tutorial
    10 relies on when attending over buses, which have no natural order."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(6, 8))
    original, _ = scaled_dot_product_attention_numpy(x, x, x)
    order = np.array([3, 1, 5, 0, 4, 2])
    permuted, _ = scaled_dot_product_attention_numpy(x[order], x[order], x[order])
    np.testing.assert_allclose(permuted, original[order], atol=1e-12)


def test_positional_encoding_shape_and_range():
    encoding = sinusoidal_positional_encoding(64, 16)
    assert encoding.shape == (64, 16)
    assert encoding.abs().max() <= 1.0 + 1e-6


def test_our_multihead_attention_matches_pytorch():
    set_seed()
    d_model, n_heads = 32, 4
    ours = MultiHeadSelfAttention(d_model, n_heads, dropout=0.0).eval()
    reference = nn.MultiheadAttention(d_model, n_heads, batch_first=True).eval()
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat(
            [ours.query_proj.weight, ours.key_proj.weight, ours.value_proj.weight]))
        reference.in_proj_bias.copy_(torch.cat(
            [ours.query_proj.bias, ours.key_proj.bias, ours.value_proj.bias]))
        reference.out_proj.weight.copy_(ours.out_proj.weight)
        reference.out_proj.bias.copy_(ours.out_proj.bias)

    x = torch.randn(2, 10, d_model)
    with torch.no_grad():
        mine, weights = ours(x, need_weights=True)
        theirs, _ = reference(x, x, x, need_weights=False)
    assert torch.allclose(mine, theirs, atol=1e-5)
    assert weights.shape == (2, n_heads, 10, 10)
    assert torch.allclose(weights.sum(-1), torch.ones(2, n_heads, 10), atol=1e-5)


def test_multihead_rejects_indivisible_dimensions():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadSelfAttention(d_model=30, n_heads=4)


# --- forecasters --------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MLPForecaster(CONTEXT, HORIZON, n_channels=CHANNELS, hidden_sizes=(16,)),
        lambda: SimpleRNNForecaster(CHANNELS, hidden_size=12, horizon=HORIZON),
        lambda: LSTMForecaster(CHANNELS, hidden_size=12, horizon=HORIZON),
        lambda: TransformerForecaster(CONTEXT, HORIZON, n_channels=CHANNELS,
                                      d_model=16, n_heads=2, n_layers=1),
    ],
)
def test_forecaster_forward_shapes(factory):
    set_seed()
    model = factory()
    output = model(torch.randn(BATCH, CONTEXT, CHANNELS))
    assert output.shape == (BATCH, HORIZON)
    assert torch.isfinite(output).all()
    assert count_parameters(model) > 0


def test_transformer_handles_a_different_context_length():
    """A sequence model must not be locked to the length it was built with."""
    set_seed()
    model = TransformerForecaster(CONTEXT, HORIZON, n_channels=CHANNELS,
                                  d_model=16, n_heads=2, n_layers=1)
    assert model(torch.randn(2, CONTEXT // 2, CHANNELS)).shape == (2, HORIZON)


def test_transformer_returns_attention_weights():
    set_seed()
    model = TransformerForecaster(CONTEXT, HORIZON, n_channels=CHANNELS,
                                  d_model=16, n_heads=2, n_layers=2)
    output, weights = model(torch.randn(2, CONTEXT, CHANNELS), need_weights=True)
    assert output.shape == (2, HORIZON)
    assert weights.shape == (2, 2, 2, CONTEXT, CONTEXT)   # batch, layer, head, q, k


def test_transformer_block_preserves_shape():
    set_seed()
    block = TransformerBlock(16, 2, dropout=0.0)
    x = torch.randn(2, 9, 16)
    out, _ = block(x)
    assert out.shape == x.shape


def test_rnn_can_return_its_hidden_states():
    set_seed()
    model = SimpleRNNForecaster(CHANNELS, hidden_size=12, horizon=HORIZON)
    output, states = model(torch.randn(BATCH, CONTEXT, CHANNELS), return_states=True)
    assert output.shape == (BATCH, HORIZON)
    assert states.shape == (BATCH, CONTEXT, 12)


# --- representation learning --------------------------------------------------


def test_masked_autoencoder_roundtrip():
    set_seed()
    model = MaskedProfileAutoencoder(length=24, n_channels=1, latent_dim=8, hidden_channels=8)
    x = torch.randn(BATCH, 24, 1)
    mask = random_mask((BATCH, 24), ratio=0.3)
    assert model(x, mask=mask).shape == (BATCH, 24, 1)
    assert model.embed(x).shape == (BATCH, 8)


def test_masking_helpers():
    mask = random_mask((100, 24), ratio=0.3, generator=torch.Generator().manual_seed(0))
    assert 0.2 < mask.float().mean() < 0.4
    blocks = block_mask((50, 24), block=6, generator=torch.Generator().manual_seed(0))
    assert (blocks.sum(dim=1) == 6).all(), "each sample hides exactly one block"
    # Each masked region must be contiguous.
    for row in blocks:
        positions = torch.where(row)[0]
        assert (positions.diff() == 1).all()


def test_masked_loss_ignores_visible_positions():
    prediction = torch.zeros(2, 4, 1)
    target = torch.ones(2, 4, 1)
    mask = torch.tensor([[True, False, False, False], [False, False, False, True]])
    # Only one position per sample is scored, and each is off by exactly 1.
    assert masked_reconstruction_loss(prediction, target, mask) == pytest.approx(1.0)


def test_masked_loss_is_zero_when_nothing_is_masked_wrongly():
    x = torch.randn(3, 6, 1)
    mask = torch.zeros(3, 6, dtype=torch.bool)
    mask[:, 0] = True
    assert masked_reconstruction_loss(x, x, mask) == pytest.approx(0.0, abs=1e-6)


def test_nt_xent_rewards_agreement():
    set_seed()
    z = torch.randn(8, 5)
    agreeing = nt_xent_loss(z, z.clone())
    disagreeing = nt_xent_loss(z, torch.randn(8, 5))
    assert agreeing < disagreeing


def test_linear_probe_freezes_its_encoder():
    set_seed()
    encoder = ProfileEncoder(length=24, n_channels=1, latent_dim=8, hidden_channels=8)
    probe = LinearProbe(encoder, latent_dim=8, n_outputs=1)
    assert all(not p.requires_grad for p in probe.encoder.parameters())
    assert all(p.requires_grad for p in probe.head.parameters())

    before = [p.clone() for p in probe.encoder.parameters()]
    output = probe(torch.randn(4, 24, 1))
    output.sum().backward()
    torch.optim.SGD(probe.parameters(), lr=1.0).step()
    for old, new in zip(before, probe.encoder.parameters(), strict=True):
        assert torch.equal(old, new), "a frozen encoder must not move"


# --- tiny GPT -----------------------------------------------------------------


def test_char_tokenizer_roundtrip():
    text = "LOG 07:15 Nordfeld 380 kV"
    tokenizer = CharTokenizer(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.vocab_size == len(set(text))


def test_tinygpt_forward_and_loss():
    set_seed()
    config = TinyGPTConfig(vocab_size=20, context=16, d_model=16, n_heads=2, n_layers=2)
    model = TinyGPT(config)
    idx = torch.randint(0, 20, (2, 16))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 16, 20)
    assert loss.item() > 0
    # An untrained model should be near the uniform-guess loss.
    assert abs(loss.item() - math.log(20)) < 0.6


def test_tinygpt_rejects_an_overlong_sequence():
    config = TinyGPTConfig(vocab_size=10, context=8, d_model=8, n_heads=2, n_layers=1)
    with pytest.raises(ValueError, match="exceeds context"):
        TinyGPT(config)(torch.randint(0, 10, (1, 9)))


def test_generation_is_causal_and_deterministic_at_zero_temperature():
    set_seed()
    config = TinyGPTConfig(vocab_size=12, context=16, d_model=16, n_heads=2, n_layers=1)
    model = TinyGPT(config)
    prompt = torch.randint(0, 12, (1, 4))
    a = generate(model, prompt.clone(), max_new_tokens=6, temperature=0.0)
    b = generate(model, prompt.clone(), max_new_tokens=6, temperature=0.0)
    assert torch.equal(a, b)
    assert a.shape == (1, 10)
    assert torch.equal(a[:, :4], prompt)


def test_top_k_and_top_p_truncate_the_distribution():
    """Truncation is per step, so test the filter itself rather than a rollout.

    (A generation can legitimately contain more than k distinct tokens: the
    top-k set is recomputed at every step as the context changes.)
    """
    from ai_power_course.models.tinygpt import _filter_logits

    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0, -2.0]])

    top_k = _filter_logits(logits.clone(), top_k=3, top_p=None)
    assert torch.isinf(top_k).sum() == 3
    assert torch.softmax(top_k, dim=-1)[0, 3:].sum() == 0.0

    top_p = _filter_logits(logits.clone(), top_k=None, top_p=0.5)
    kept = torch.isfinite(top_p).sum().item()
    assert 1 <= kept < logits.shape[1]
    assert torch.softmax(top_p, dim=-1).sum() == pytest.approx(1.0)

    # The most likely token always survives, whatever p is.
    assert torch.isfinite(_filter_logits(logits.clone(), None, 0.01))[0, 0]


# --- training loop ------------------------------------------------------------


def test_standardizer_fits_on_training_data_only():
    train = np.random.default_rng(0).normal(10, 2, size=(100, 3))
    scaler = Standardizer().fit(train)
    transformed = scaler.transform(train)
    np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(transformed.std(axis=0), 1, atol=1e-3)
    np.testing.assert_allclose(scaler.inverse_transform(transformed), train, atol=1e-3)


def test_standardizer_requires_fitting():
    with pytest.raises(RuntimeError, match="fitted"):
        Standardizer().transform(np.zeros((2, 2)))


def test_training_reduces_the_loss_and_restores_the_best_weights():
    set_seed()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 6)).astype(np.float32)
    y = (x @ rng.normal(size=(6, 1))).astype(np.float32)
    model = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 1))

    history = train_model(
        model,
        make_loader(x[:200], y[:200], batch_size=32, shuffle=True),
        make_loader(x[200:], y[200:], batch_size=64),
        config=TrainConfig(epochs=25, learning_rate=1e-2, verbose=False, patience=None),
    )
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.best_val_loss == min(history.val_loss)
    assert history.seconds > 0
    assert predict(model, x).shape == (256, 1)


def test_early_stopping_triggers():
    set_seed()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 4)).astype(np.float32)
    y = rng.normal(size=(64, 1)).astype(np.float32)   # pure noise: cannot improve
    model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 1))
    history = train_model(
        model,
        make_loader(x, y, batch_size=8, shuffle=True),
        make_loader(x, y, batch_size=32),
        config=TrainConfig(epochs=200, learning_rate=1e-2, patience=3, verbose=False),
    )
    assert len(history.train_loss) < 200, "early stopping should have fired"
