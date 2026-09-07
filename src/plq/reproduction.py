"""Provenance and fixed-model experimental histogram comparisons.

Running paper parameters is never automatically promoted to reproducing an
experiment. Statistical agreement alone also does not establish reproduction.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import xlogy
from .numerics import integer


def content_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence_record(category, *, sources=(), assumptions=()):
    if category not in ("simulation", "paper_parameter_reproduction", "synthetic_validation", "experimental_data_comparison"):
        raise ValueError("Unsupported evidence category; experimental reproduction requires external protocol review")
    return {"category": category, "experimental_reproduction": False,
            "sources": list(sources), "assumptions": list(assumptions)}


def _keys(data, allowed, required=()):
    if not isinstance(data, dict) or set(data)-set(allowed) or set(required)-set(data):
        raise ValueError(f"Invalid comparison fields; allowed={sorted(allowed)}, required={sorted(required)}")


def compare_experiment(manifest, *, base_directory=".", bootstrap_samples=2000, seed=0):
    """Compare complete, independent trial categories with a fixed optical model.

    Dataset: {origin, shots, bins: [{outcome: [...], count: N}], other_count}.
    Every trial is counted, including failure/vacuum and unlisted outcomes in
    'other'. Overlapping coincidence windows and fitted models are unsupported.
    Origin/protocol are supplied by the caller, not independently authenticated.
    """
    required = {"dataset", "dataset_sha256", "scenario", "observable", "protocol", "calibration", "sources"}
    _keys(manifest, required, required)
    _keys(manifest["protocol"], {"trial_definition", "selection", "independent_trials", "same_observable", "fixed_parameters"},
          {"trial_definition", "selection", "independent_trials", "same_observable", "fixed_parameters"})
    protocol = manifest["protocol"]
    if any(protocol[k] is not True for k in ("independent_trials", "same_observable", "fixed_parameters")):
        raise ValueError("Comparison requires independent trials, the same observable and fixed parameters")
    if protocol["selection"] != "all_trials" or not isinstance(protocol["trial_definition"], str) or not protocol["trial_definition"].strip():
        raise ValueError("Supply a trial definition and selection=all_trials; postselected-only data is unsupported")
    if not isinstance(manifest["calibration"], dict) or not manifest["calibration"]:
        raise ValueError("Describe calibration sources and missing calibration in a nonempty object")
    if not isinstance(manifest["sources"], list) or not manifest["sources"] or any(not isinstance(s, str) or not s.strip() for s in manifest["sources"]):
        raise ValueError("Dataset/protocol source references are required")
    root = Path(base_directory)
    dataset_path, scenario_path = root/manifest["dataset"], root/manifest["scenario"]
    actual_hash = content_hash(dataset_path)
    if manifest["dataset_sha256"] != actual_hash:
        raise ValueError("Dataset SHA256 mismatch; verify the data before comparison")
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    _keys(data, {"origin", "shots", "bins", "other_count"}, {"origin", "shots", "bins", "other_count"})
    if data["origin"] not in ("experimental", "synthetic"):
        raise ValueError("Dataset origin must explicitly be experimental or synthetic")
    shots = integer(data["shots"], "shots", 1)
    bins, counts = [], []
    for item in data["bins"]:
        _keys(item, {"outcome", "count"}, {"outcome", "count"})
        outcome = tuple(integer(v, "outcome") for v in item["outcome"])
        if outcome in bins:
            raise ValueError("Repeated dataset outcome")
        bins.append(outcome)
        counts.append(integer(item["count"], "count"))
    if not bins:
        raise ValueError("At least one outcome bin is required")
    counts.append(integer(data["other_count"], "other_count"))
    if sum(counts) != shots:
        raise ValueError("All bin counts plus other_count must equal shots; rejected trials cannot disappear")
    from .cli import optical_scenario
    config = json.loads(scenario_path.read_text(encoding="utf-8"))
    if config.get("backend", "density") != "density":
        raise ValueError("Experimental comparison needs exact density predictions, not Monte Carlo estimates")
    prediction = optical_scenario(config)
    if prediction["source_omitted_probability"] or prediction["bath_omitted_probability"]:
        raise ValueError("Comparison requires no omitted source/bath probability; specify a complete finite model")
    if manifest["observable"] == "photon_occupation":
        values = {tuple(v["occupation"]): v["probability"] for v in prediction["probabilities"]}
    elif manifest["observable"] == "detector_counts":
        if "detector_probabilities" not in prediction:
            raise ValueError("detector_counts requires detectors in the scenario")
        values = {tuple(v["counts"]): v["probability"] for v in prediction["detector_probabilities"]}
    else:
        raise ValueError("observable must be photon_occupation or detector_counts")
    if any(len(b) != config["modes"] for b in bins):
        raise ValueError("Dataset outcome width must match circuit modes")
    if abs(sum(values.values())-1) > 1e-10:
        raise ValueError("Prediction must include all trial probability")
    probs = [values.get(b, 0.) for b in bins]
    probs.append(max(0., sum(p for b, p in values.items() if b not in bins)))
    probs = np.asarray(probs)
    # Only complex128 trace roundoff is normalized, and its size is reported.
    residual = float(abs(probs.sum()-1))
    probs /= probs.sum()
    observations = np.asarray(counts)
    expected = shots*probs
    impossible = bool(np.any((expected == 0) & (observations > 0)))
    bootstrap_samples = integer(bootstrap_samples, "bootstrap_samples", 1)
    seed = integer(seed, "seed")
    if bootstrap_samples > 100_000 or len(probs) > 10_000:
        raise ValueError("Comparison exceeds bootstrap/bin budget")
    statistic, p_value = None, 0.
    if not impossible:
        positive = expected > 0
        e, n = expected[positive], observations[positive]
        statistic = float(max(0., 2*np.sum(xlogy(n, n)-xlogy(n, e))))
        rng, extreme = np.random.default_rng(seed), 0
        # Stream bootstrap trials: storage does not grow with replicate count.
        for _ in range(bootstrap_samples):
            draw = rng.multinomial(shots, probs)[positive]
            deviance = max(0., 2*float(np.sum(xlogy(draw, draw)-xlogy(draw, e))))
            extreme += deviance >= statistic-1e-12
        p_value = (extreme+1)/(bootstrap_samples+1)
    return {"format": "plq.experimental-comparison.v1",
            "evidence": evidence_record("experimental_data_comparison" if data["origin"] == "experimental" else "synthetic_validation",
                sources=manifest["sources"], assumptions=["Protocol and dataset origin are user declarations, not independently authenticated.",
                  "Fixed-model multinomial bootstrap; no fitting or calibration uncertainty marginalization.",
                  "Agreement is not proof of experimental reproduction."]),
            "manifest": manifest, "dataset_sha256": actual_hash, "scenario_sha256": content_hash(scenario_path),
            "prediction_versions": prediction["versions"], "shots": shots,
            "bins": [{"outcome": list(b) if b is not None else "other", "observed": int(n),
                      "expected": float(e)} for b, n, e in zip([*bins, None], observations, expected)],
            "deviance": statistic, "impossible_observed_event": impossible, "bootstrap_p_value": p_value,
            "bootstrap_samples": bootstrap_samples, "seed": seed, "prediction_trace_roundoff": residual}
