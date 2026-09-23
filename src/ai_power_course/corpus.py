"""A small, self-contained power-engineering text corpus.

Tutorial 07 trains a character-level language model on this; tutorial 08 uses
the prose sections as the document store for a retrieval system.

**Provenance.** Everything here was written for this course and is covered by
the repository's MIT licence. The explanatory prose in :data:`HANDBOOK` is
genuine technical writing. The operator logs, disturbance reports and asset
records produced by :func:`generate_corpus` are **synthetic**, generated from
templates with deterministic randomness — they are not extracts from any real
control-room system, and the notebooks say so.

Using a self-written corpus avoids the licensing problem entirely: real
operator logs are confidential, and scraped technical documentation usually
carries terms that forbid redistribution. The cost is that the corpus is small,
which is exactly why tutorial 07 trains a *tiny* model and is explicit that the
gap to a real LLM is data and compute, not a different idea.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

__all__ = [
    "HANDBOOK",
    "HANDBOOK_SECTIONS",
    "generate_corpus",
    "corpus_path",
    "write_corpus",
    "EVENT_CLASSES",
    "generate_event_dataset",
]


# --- hand-written technical prose --------------------------------------------
# Short, self-contained sections. Tutorial 08 retrieves over these, so each one
# is written to stand alone and to answer a question a student might actually
# ask.

HANDBOOK_SECTIONS: dict[str, str] = {
    "per-unit system": """
        The per-unit system expresses electrical quantities as fractions of a
        chosen base. A base apparent power S_base and a base voltage V_base are
        selected for each voltage level; base current and base impedance follow
        as I_base = S_base / (sqrt(3) V_base) and Z_base = V_base^2 / S_base.
        Dividing every quantity by its base removes the voltage level from the
        numbers, so a 380 kV line and a 20 kV feeder have comparable impedances
        of a few hundredths per unit. Transformer ratios largely disappear,
        which is the main practical reason the convention is universal in power
        flow software. A bus voltage of 1.02 per unit means two percent above
        nominal regardless of whether nominal is 400 volts or 400 kilovolts.
        """,
    "power flow": """
        The AC power flow problem solves for the complex voltage at every bus
        given the injections and the network admittance matrix. The governing
        equation is S = V .* conj(Y V), where S is the vector of complex nodal
        injections, V the vector of complex bus voltages and Y the bus
        admittance matrix. Because the equations are non-linear in the voltage
        magnitudes and angles, they are solved iteratively, most commonly by the
        Newton-Raphson method. Each bus is classified as PQ, where active and
        reactive injections are known, PV, where active power and voltage
        magnitude are known, or slack, where voltage magnitude and angle are
        fixed and the injection is whatever balances the system. Convergence is
        not guaranteed: heavily loaded systems near the nose of the PV curve, or
        systems that have been split by an outage, may have no solution at all.
        """,
    "voltage limits": """
        Operational voltage limits protect equipment and customer appliances.
        Transmission planning commonly uses a band of 0.95 to 1.05 per unit,
        while EN 50160 permits plus or minus ten percent at the point of common
        coupling in low voltage networks for 95 percent of weekly ten-minute
        averages. Undervoltage is typically caused by heavy load far from a
        source; overvoltage in distribution networks is increasingly caused by
        distributed generation exporting power along a feeder with appreciable
        resistance. Because the R/X ratio of low voltage cables is high, active
        power export raises voltage much more strongly there than in a
        transmission network, which is why photovoltaic hosting capacity is
        frequently limited by voltage rather than by thermal rating.
        """,
    "thermal rating": """
        A line or transformer is thermally limited by the temperature its
        conductors and insulation can tolerate. The static rating is a current
        derived from a conservative assumption about ambient temperature, wind
        speed and solar radiation. Because the actual cooling conditions are
        usually better than assumed, dynamic line rating can release substantial
        additional capacity, particularly on windy days, which is also when wind
        generation is highest. Loading is normally reported as a percentage of
        the rating, and a value above one hundred percent is an overload. Short
        overloads are not immediately damaging; the constraint is thermal energy
        over time, so ratings are often specified for several durations.
        """,
    "N-1 criterion": """
        The N-1 criterion requires that the system remain within all operational
        limits after the loss of any single component. It is a deterministic
        reliability standard and the backbone of European transmission planning
        and operation. Compliance is assessed by contingency analysis: a power
        flow is solved for the base case and then for every credible single
        outage, and the resulting voltages and branch loadings are checked
        against their limits. For a network with several thousand branches this
        means several thousand power flows, which is why fast approximations and
        machine learning surrogates for contingency screening are an active
        research area.
        """,
    "state estimation": """
        State estimation reconstructs the most likely system state from
        redundant and noisy measurements. The classical formulation is weighted
        least squares: minimise the weighted sum of squared differences between
        measured quantities and the values implied by an estimated state, subject
        to the network model. Redundancy is essential, both to reduce the effect
        of measurement noise and to allow bad data detection. A system is
        observable if the available measurements are sufficient to determine the
        state uniquely; unobservable islands must be handled by pseudo
        measurements, typically load forecasts. Phasor measurement units provide
        direct, time-synchronised measurements of voltage angle, which changes
        the structure of the problem considerably.
        """,
    "load forecasting": """
        Short-term load forecasting predicts demand from minutes to a few days
        ahead and underpins unit commitment, market bidding and congestion
        management. Demand is driven by the time of day, the day of the week,
        holidays, and weather, principally temperature. The relationship with
        temperature is V-shaped: heating demand rises below a comfort threshold
        of roughly fifteen degrees Celsius and cooling demand rises above
        roughly twenty-two degrees, so a single linear coefficient averages two
        opposing effects into nothing. Strong baselines are persistence and
        seasonal naive forecasts; a model that does not clearly beat the value
        from the same hour of the previous week has not demonstrated anything.
        """,
    "renewable forecasting": """
        Photovoltaic generation is bounded below by zero and is exactly zero
        whenever the sun is below the horizon, so any forecast that predicts
        non-zero output at night is physically wrong regardless of its error
        metric. Clipping predictions at zero and forcing night hours to zero is
        free accuracy and removes an entire class of impossible states. Wind
        power follows a turbine power curve: output is zero below the cut-in
        speed of about three metres per second, rises approximately with the
        cube of wind speed, saturates at rated speed around twelve metres per
        second, and is curtailed above the cut-out speed near twenty-five metres
        per second. Aggregating many sites smooths the cut-out discontinuity
        because the wind speed differs across a country.
        """,
    "residual load": """
        Residual load is demand minus non-dispatchable generation, usually wind
        and photovoltaic. It is the quantity the conventional fleet and the
        market must actually serve, and it is more volatile than demand alone
        because it inherits weather variability. As installed renewable capacity
        grows, residual load develops a pronounced midday depression on sunny
        days, the shape often called the duck curve, and can become negative in
        periods of low demand and high wind. Negative residual load is the
        principal driver of negative day-ahead prices, which occur when
        inflexible generation and must-run units would rather pay to keep
        producing than shut down and restart.
        """,
    "day-ahead market": """
        The day-ahead market clears hourly or quarter-hourly products for the
        following delivery day, typically by a uniform-price auction at midday.
        The clearing price is set by the marginal unit, so the price curve
        resembles the merit order of the fleet: cheap when residual load is low
        and steep when expensive peaking plant is required. Prices can be
        negative when renewable generation exceeds demand and curtailment is
        more expensive than paying consumers to absorb energy. Forecast errors
        in load or renewable generation are settled in the intraday and balancing
        markets, which is why forecast accuracy has a direct and measurable
        economic value.
        """,
    "frequency control": """
        System frequency is the instantaneous indicator of the balance between
        generation and demand. A surplus of generation raises frequency and a
        deficit lowers it, with the rate of change determined by the inertia of
        the rotating masses connected to the system. Frequency containment
        reserve responds automatically within seconds to arrest a deviation,
        frequency restoration reserve returns frequency to nominal over minutes,
        and replacement reserve restores the faster products. Converter-connected
        generation provides no inherent inertia, so as its share grows the rate
        of change of frequency after a disturbance increases, and synthetic
        inertia or faster reserve products become necessary.
        """,
    "graph representation": """
        A power network is naturally a graph: buses are nodes and branches are
        edges. Representing it this way matters for machine learning because a
        fixed-width model trained on a thirty-bus network cannot even be
        evaluated on a hundred-and-eighteen-bus one, while a message-passing
        model shares its parameters across every node and edge and therefore
        applies unchanged to a network it has never seen. Node features
        typically include the active and reactive injection, the voltage
        magnitude and angle, and the bus type; edge features carry the series
        resistance and reactance, the shunt susceptance and the transformer tap
        ratio. Expressing all of these in per unit is what makes the features
        comparable across networks of different voltage levels.
        """,
}

HANDBOOK = "\n\n".join(
    f"{title.upper()}\n{textwrap.fill(' '.join(body.split()), width=78)}"
    for title, body in HANDBOOK_SECTIONS.items()
)


# --- synthetic structured records --------------------------------------------

_SUBSTATIONS = ("Nordfeld", "Ostkamp", "Lindenau", "Sudwerk", "Altbach", "Rheinhof",
                "Kirchberg", "Weidental", "Hochmoor", "Steinbruck")

#: Voltage level -> plausible (min, max) load in MW and (min, max) equipment
#: rating in MVA. A 20 kV feeder does not carry 400 MW, and a corpus that says
#: it does would teach a student the wrong order of magnitude even though every
#: individual sentence is grammatical.
_LEVELS: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {
    20: ((2.0, 25.0), (5.0, 40.0)),
    30: ((4.0, 40.0), (8.0, 63.0)),
    110: ((20.0, 180.0), (31.5, 250.0)),
    220: ((80.0, 450.0), (100.0, 600.0)),
    380: ((200.0, 1200.0), (300.0, 1200.0)),
}
_VOLTAGES = tuple(_LEVELS)
_EQUIPMENT = ("circuit breaker", "disconnector", "power transformer", "overhead line",
              "underground cable", "capacitor bank", "voltage regulator", "busbar")
_CAUSES = ("lightning strike", "vegetation contact", "insulation failure",
           "protection maloperation", "overload", "storm damage", "animal contact",
           "equipment ageing", "scheduled maintenance", "cable joint failure")
_WEATHER = ("clear", "overcast", "light rain", "heavy rain", "thunderstorm",
            "snow", "high wind", "fog")
_ACTIONS = ("auto-reclose successful", "switched to reserve feeder",
            "load shed on feeder", "generation redispatched",
            "tap changer adjusted", "capacitor bank connected",
            "line returned to service", "isolated for repair")


def _log_entry(rng: np.random.Generator) -> str:
    substation = rng.choice(_SUBSTATIONS)
    voltage = int(rng.choice(_VOLTAGES))
    hour, minute = int(rng.integers(0, 24)), int(rng.integers(0, 60))
    load = round(float(rng.uniform(*_LEVELS[voltage][0])), 1)
    voltage_pu = round(float(rng.normal(1.01, 0.025)), 3)
    loading = round(float(rng.uniform(18, 118)), 1)
    return (
        f"LOG {hour:02d}:{minute:02d} {substation} {voltage} kV | "
        f"load {load} MW | voltage {voltage_pu} pu | loading {loading} percent | "
        f"{rng.choice(_ACTIONS)}."
    )


def _disturbance_report(rng: np.random.Generator) -> str:
    substation = rng.choice(_SUBSTATIONS)
    voltage = int(rng.choice(_VOLTAGES))
    equipment = rng.choice(_EQUIPMENT)
    cause = rng.choice(_CAUSES)
    weather = rng.choice(_WEATHER)
    duration = int(rng.integers(1, 320))
    interrupted = int(rng.integers(0, 400) * max(voltage // 20, 1))
    energy = round(duration * interrupted * 0.0009, 1)
    return (
        f"DISTURBANCE REPORT\n"
        f"  location    : {substation} {voltage} kV\n"
        f"  equipment   : {equipment}\n"
        f"  cause       : {cause}\n"
        f"  weather     : {weather}\n"
        f"  duration    : {duration} minutes\n"
        f"  customers   : {interrupted}\n"
        f"  energy not supplied : {energy} MWh\n"
        f"  remedial action : {rng.choice(_ACTIONS)}."
    )


def _asset_record(rng: np.random.Generator) -> str:
    substation = rng.choice(_SUBSTATIONS)
    voltage = int(rng.choice(_VOLTAGES))
    equipment = rng.choice(_EQUIPMENT)
    year = int(rng.integers(1968, 2024))
    rating = round(float(rng.uniform(*_LEVELS[voltage][1])), 1)
    return (
        f"ASSET {substation}-{voltage}-{equipment.replace(' ', '_')} | "
        f"commissioned {year} | rated {rating} MVA | "
        f"condition index {round(float(rng.uniform(0.2, 1.0)), 2)} | "
        f"next inspection {int(rng.integers(2026, 2032))}."
    )


def generate_corpus(
    n_logs: int = 1400, n_reports: int = 220, n_assets: int = 320, seed: int = 20260101
) -> str:
    """Build the full corpus: hand-written prose plus synthetic records.

    The prose is repeated a few times so that a character-level model trained on
    the whole file sees enough of it to learn technical vocabulary; the records
    supply the rigid structure that makes a tiny model's progress visible.
    """
    rng = np.random.default_rng(seed)
    # The handbook appears three times, and all three copies sit near the FRONT.
    #
    # They used to be appended at the end (`parts += ["", HANDBOOK, "",
    # HANDBOOK]`), which put both repeats inside the final 10% of the file --
    # exactly the slice Tutorial 07 holds out for validation. The result was a
    # validation set that was mostly a verbatim copy of training text: 66% of
    # sampled 65-character validation windows occurred character-for-character
    # in the training split, so the reported validation perplexity measured
    # recall of text the model had already seen, not generalisation.
    #
    # The repetition itself is wanted -- a character model needs to see
    # technical vocabulary more than once -- so the copies stay. They just
    # belong where the training data is.
    parts: list[str] = [
        "POWER SYSTEMS HANDBOOK",
        "",
        HANDBOOK,
        "",
        HANDBOOK,
        "",
        HANDBOOK,
        "",
        "SYNTHETIC OPERATIONAL RECORDS",
        "The entries below are generated, not measured. They exist so that a small",
        "language model has structured text to learn from. No real substation, asset",
        "or disturbance is described.",
        "",
    ]
    # INTERLEAVE the three record types instead of writing them in blocks.
    #
    # Tutorial 07 holds out the last 10% of this file. With the records grouped
    # by type, that slice was 233 ASSET records and nothing else -- zero logs,
    # zero disturbance reports -- against a training mix of 1,400 logs, 220
    # reports and 87 assets. Validation was a different KIND of text from
    # training, and asset records are the most rigid template of the three, so
    # the reported perplexity measured "can it finish an asset record" rather
    # than "did it generalise". Fixing the earlier duplication bug made this
    # visible: validation perplexity IMPROVED from 3.59 to 1.35, which is the
    # wrong direction for removing a leak and is what exposed the confound.
    #
    # Shuffling the records together makes any contiguous slice a
    # representative sample of the record distribution. The handbook stays at
    # the front, so the prose is training-only by construction -- that is a
    # deliberate choice and the tutorial says so.
    records = (
        [_log_entry(rng) for _ in range(n_logs)]
        + [_disturbance_report(rng) + "\n" for _ in range(n_reports)]
        + [_asset_record(rng) for _ in range(n_assets)]
    )
    parts += [records[i] for i in rng.permutation(len(records))]
    return "\n".join(parts)


def corpus_path() -> Path:
    from .config import SAMPLE_DIR

    return SAMPLE_DIR / "power_systems_corpus.txt"


def write_corpus(path: Path | None = None, **kwargs) -> Path:
    """(Re)generate the committed corpus file."""
    target = path or corpus_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(generate_corpus(**kwargs), encoding="utf-8")
    return target


# --- a labelled classification dataset ---------------------------------------
# Tutorial 08 needs a supervised text task. Real incident databases are
# confidential, so this one is generated the same way as the logs above: from
# templates, deterministically, and labelled by construction.
#
# The phrasings are deliberately varied and the classes deliberately share
# vocabulary ("the line", "the feeder", "voltage") so that a bag-of-words model
# cannot solve it from a single keyword. It is still an easy task — any
# synthetic dataset built from templates is — and the notebook says so before
# drawing conclusions from the accuracy.

EVENT_CLASSES: tuple[str, ...] = ("routine", "overload", "voltage_violation", "outage")

_BRANCHES = ("overhead line", "underground cable", "power transformer")
_SWITCHGEAR = ("circuit breaker", "disconnector", "feeder bay")

# Placeholders are *typed* so that every generated sentence is physically
# coherent: {vm_low} is always an undervoltage, {vm_high} always an
# overvoltage, {branch} is always something that carries power flow, and
# {small} is a forecast-error-sized number rather than a system-sized one.
# Nonsense like "raised the busbar to 0.91 per unit" would teach a student the
# wrong physics even while training a perfectly good classifier.
_EVENT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "routine": (
        "Scheduled inspection completed on the {branch} at {sub}; no anomalies found.",
        "Routine switching at {sub} carried out as planned, system remained within limits.",
        "Tap changer on the power transformer at {sub} operated normally during the morning ramp.",
        "All measurements at {sub} nominal; {flow} MW on the {branch}, well inside its rating.",
        "Periodic protection test at {sub} passed; the {switch} was returned to service.",
        "Reactive compensation at {sub} adjusted as part of planned voltage management.",
        "Demand at {sub} followed the forecast to within {small} MW; no operator action required.",
        "Maintenance window at {sub} closed on schedule with the {branch} back in service.",
    ),
    "overload": (
        "The {branch} at {sub} reached {loading} percent of its rating during the evening peak.",
        "Thermal limit exceeded on the {branch} serving {sub}; {loading} percent "
        "for {mins} minutes.",
        "Congestion at {sub} required redispatch after the {branch} passed {loading} percent.",
        "Loading on the {branch} climbed to {loading} percent; curtailment applied upstream.",
        "Post-contingency flow on the {branch} at {sub} would exceed the rating "
        "by {flow} MW.",
        "Sustained current above the continuous rating on the {branch} at {sub} "
        "for {mins} minutes.",
        "Ampacity of the {branch} feeding {sub} was the binding constraint during the cold spell.",
        "Dynamic rating of the {branch} at {sub} was insufficient; {loading} percent recorded.",
    ),
    "voltage_violation": (
        "Bus voltage at {sub} fell to {vm_low} per unit under heavy load, below the lower band.",
        "Overvoltage of {vm_high} per unit recorded at {sub} during high photovoltaic export.",
        "Voltage at {sub} drifted outside the admissible band for {mins} minutes.",
        "Reverse power flow at {sub} pushed the feeder end voltage up to {vm_high} per unit.",
        "Undervoltage alarm at {sub}; {vm_low} per unit measured with the {branch} in service.",
        "Steady-state voltage at {sub} outside the statutory range; tap changer at its limit.",
        "Midday export from distributed generation raised {sub} to {vm_high} per unit.",
        "Voltage at the remote end of the {branch} from {sub} sagged to {vm_low} per unit.",
    ),
    "outage": (
        "The {branch} at {sub} tripped and was isolated; supply restored via the reserve feeder.",
        "Unplanned disconnection of the {branch} at {sub} after a protection operation.",
        "Loss of supply at {sub} for {mins} minutes following a fault on the {branch}.",
        "The {branch} serving {sub} was taken out of service after an insulation failure.",
        "Automatic reclosing on the {branch} at {sub} was unsuccessful; the circuit stayed open.",
        "Fault on the {branch} near {sub} interrupted {flow} MW of demand.",
        "Busbar protection at {sub} operated and disconnected the {switch}.",
        "Storm damage to the {branch} at {sub} left customers without supply for {mins} minutes.",
    ),
}


def generate_event_dataset(n_per_class: int = 220, seed: int = 20260102):
    """Synthetic (text, label) pairs for the tutorial 08 classification task.

    Returns ``(texts, labels)`` where ``labels`` index :data:`EVENT_CLASSES`.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    texts: list[str] = []
    labels: list[int] = []
    for index, event_class in enumerate(EVENT_CLASSES):
        templates = _EVENT_TEMPLATES[event_class]
        for _ in range(n_per_class):
            template = templates[int(rng.integers(len(templates)))]
            texts.append(
                template.format(
                    sub=rng.choice(_SUBSTATIONS),
                    branch=rng.choice(_BRANCHES),
                    switch=rng.choice(_SWITCHGEAR),
                    flow=round(float(rng.uniform(5, 320)), 1),
                    small=round(float(rng.uniform(0.4, 25.0)), 1),
                    loading=round(float(rng.uniform(101, 145)), 1),
                    vm_low=round(float(rng.uniform(0.86, 0.949)), 3),
                    vm_high=round(float(rng.uniform(1.051, 1.13)), 3),
                    mins=int(rng.integers(3, 240)),
                )
            )
            labels.append(index)
    order = rng.permutation(len(texts))
    return [texts[i] for i in order], np.asarray(labels)[order]
