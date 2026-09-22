"""Pre-generate the Mini-GridFM operating points and cache them.

Tutorial 10 samples these itself, which takes a couple of minutes. Running this
script first caches them so the notebook starts instantly — useful when teaching
live, and how CI avoids paying for the sampling twice.

    uv run python scripts/generate_grid_data.py

Nothing is downloaded: the networks ship with pandapower.

Edit the configuration block to change what is generated; there is no
command-line interface on purpose.
"""

from __future__ import annotations

import time

import numpy as np

from ai_power_course.config import ARTIFACT_DIR
from ai_power_course.grid.networks import HELDOUT_NETWORKS, PRETRAIN_NETWORKS, network_summary
from ai_power_course.grid.sampling import SamplingConfig, build_grid_dataset

# --- configuration -----------------------------------------------------------

#: Operating points per network.
N_SAMPLES = 400

#: Networks to sample. Pretraining grids and held-out grids are cached together;
#: tutorial 10 keeps them apart.
NETWORKS = PRETRAIN_NETWORKS + HELDOUT_NETWORKS

SAMPLING = SamplingConfig(
    n_samples=N_SAMPLES,
    load_scale=(0.6, 1.3),
    load_spread=0.15,
    sgen_scale=(0.0, 1.6),
    gen_scale=(0.85, 1.15),
    outage_probability=0.15,
    seed=20260101,
)

OUTPUT = ARTIFACT_DIR / "grid_operating_points.npz"

# -----------------------------------------------------------------------------


def main(networks: tuple[str, ...] = NETWORKS, config: SamplingConfig = SAMPLING,
         output=OUTPUT) -> None:
    print("Networks in the catalogue:\n")
    print(network_summary(networks).to_string())
    print(f"\nSampling {config.n_samples} operating points per network...\n")

    started = time.perf_counter()
    dataset = build_grid_dataset(networks, config, verbose=True)
    elapsed = time.perf_counter() - started

    print(f"\n{len(dataset)} states in {elapsed:.1f} s "
          f"({elapsed / max(len(dataset), 1) * 1000:.0f} ms per state)")
    overloaded = sum(point.n_overloaded > 0 for point in dataset)
    print(f"  overloaded states : {overloaded} ({overloaded / len(dataset):.1%})")
    print(f"  with an outage    : {sum(point.has_outage for point in dataset)}")
    print(f"  bus counts        : {sorted({point.n_buses for point in dataset})}")

    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "network": np.array([point.network for point in dataset]),
        "n_buses": np.array([point.n_buses for point in dataset]),
        "max_line_loading": np.array([point.max_line_loading for point in dataset]),
        "n_overloaded": np.array([point.n_overloaded for point in dataset]),
        "has_outage": np.array([point.has_outage for point in dataset]),
        "load_scale": np.array([point.load_scale for point in dataset]),
    }
    # Variable-size arrays are stored per sample; object arrays keep it simple
    # and the dataset is small enough that efficiency is not the concern.
    for key in ("node_features", "edge_index", "edge_features"):
        payload[key] = np.array(
            [point.graph[key].numpy() for point in dataset], dtype=object
        )
    np.savez_compressed(output, **payload, allow_pickle=True)
    print(f"\ncached to {output} ({output.stat().st_size / 1e6:.1f} MB)")
    print("Tutorial 10 regenerates these itself if the cache is absent, so this")
    print("script is a convenience rather than a prerequisite.")


if __name__ == "__main__":
    main()
