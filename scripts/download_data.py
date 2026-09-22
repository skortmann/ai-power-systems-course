"""Download and cache the real Open Power System Data time series.

Run it once; every notebook then picks the real data up automatically:

    uv run python scripts/download_data.py

Nothing in the course *requires* this. The committed synthetic dataset runs all
ten tutorials offline. Downloading the real series is how you find out which of
the course's conclusions survive contact with measurements — which is exercise 3
of tutorial 01, and worth doing.

Edit the configuration block below to change what happens; there is no
command-line interface on purpose.
"""

from __future__ import annotations

from ai_power_course.data import (
    OPSD_CITATION,
    OPSD_URL,
    download_opsd,
    load_energy_data,
)

# --- configuration -----------------------------------------------------------

#: Re-download even if the cache already exists.
FORCE = False

#: Print a summary and a synthetic/real comparison after downloading.
VERIFY = True

# -----------------------------------------------------------------------------


def main(force: bool = FORCE, verify: bool = VERIFY) -> None:
    print("Open Power System Data — time series (Germany)")
    print(f"  source : {OPSD_URL}")
    print("  size   : ~130 MB download, ~2 MB cached after column selection")
    print("\nThis data is NOT redistributed with the course. By downloading it you")
    print("accept the publisher's terms and take on the attribution obligation:\n")
    print(f"  {OPSD_CITATION}\n")
    print("See data/README.md for why it is not committed.\n")

    path = download_opsd(force=force)
    print(f"cached at {path} ({path.stat().st_size / 1e6:.1f} MB)")

    if not verify:
        return

    real = load_energy_data(source="opsd")
    print("\n" + real.describe())

    frame = real.frame
    years = (frame.index[-1] - frame.index[0]).days / 365.25
    print(f"\nannual energy over {years:.1f} years:")
    for column, label in [("load_mw", "load"), ("pv_mw", "PV"), ("wind_mw", "wind")]:
        if column in frame:
            print(f"  {label:5s} {frame[column].sum() / 1e6 / years:7.1f} TWh/a")

    synthetic = load_energy_data(source="synthetic").frame
    print("\nsynthetic vs. real, for calibration:")
    print(f"{'quantity':24s} {'synthetic':>12s} {'real':>12s}")
    rows = [
        ("mean load [MW]", synthetic.load_mw.mean(), frame.load_mw.mean()),
        ("peak load [MW]", synthetic.load_mw.max(), frame.load_mw.max()),
        ("mean PV [MW]", synthetic.pv_mw.mean(), frame.pv_mw.mean()),
        ("mean wind [MW]", synthetic.wind_mw.mean(), frame.wind_mw.mean()),
    ]
    if "price_eur_mwh" in frame:
        rows.append(
            ("negative price hours [%]",
             100 * (synthetic.price_eur_mwh < 0).mean(),
             100 * (frame.price_eur_mwh < 0).mean())
        )
    for label, synthetic_value, real_value in rows:
        print(f"{label:24s} {synthetic_value:12,.1f} {real_value:12,.1f}")

    print("\nFrom now on load_energy_data() returns the real data by default.")
    print("Force the simulation with load_energy_data(source='synthetic').")


if __name__ == "__main__":
    main()
