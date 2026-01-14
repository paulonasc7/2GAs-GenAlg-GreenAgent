#!/usr/bin/env python3
"""
Run the CVRP GAlib example with a configurable set of GA parameters.

Edit the CONFIG dict below to change operators and parameters without
recompiling the C++ code. This script invokes ./cvrp with the right
arguments and leaves the outputs for analyze_cvrp_ga.py to read:
- cvrp_stats.tsv
- cvrp_summary.txt
- cvrp_best_routes.txt
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parent
EXE = ROOT / "cvrp"

# Tweak these values as needed.
CONFIG = {
    "crossover": "order",       # order | onepoint | twopoint
    "selector": "roulette",     # roulette | tournament | rank
    "popsize": 100,
    "generations": 200,
    "pmut": 0.03,
    "pcross": 0.9,
    "nbest": 1,
    "nconv": 20,
    "score_frequency": 1,
    "flush_frequency": 20,
    "elitism": True,
    "seed": 0,                  # 0 means random seed; set >0 for reproducible runs
}


KEY_MAP = {
    "ngen": "generations",
    "generations": "generations",
    "popsize": "popsize",
    "pmut": "pmut",
    "pcross": "pcross",
    "nbest": "nbest",
    "nconv": "nconv",
    "sfreq": "score_frequency",
    "score_frequency": "score_frequency",
    "ffreq": "flush_frequency",
    "flush_frequency": "flush_frequency",
    "el": "elitism",
    "elitism": "elitism",
    "crossover": "crossover",
    "selector": "selector",
    "seed": "seed",
}


def parse_bool(val: str) -> bool:
    return val.lower() in {"1", "true", "yes", "on"}


def parse_overrides(argv: list[str]) -> dict:
    cfg = CONFIG.copy()
    i = 1
    while i < len(argv):
        key = argv[i].lstrip("-")
        if key not in KEY_MAP:
            raise SystemExit(f"Unknown parameter '{argv[i]}'. Allowed: {sorted(KEY_MAP)}")
        if i + 1 >= len(argv):
            raise SystemExit(f"Missing value for '{argv[i]}'")
        val = argv[i + 1]
        mapped = KEY_MAP[key]
        if mapped in {"crossover", "selector"}:
            cfg[mapped] = val
        elif mapped == "elitism":
            cfg[mapped] = parse_bool(val)
        elif mapped == "generations":
            cfg[mapped] = int(val)
        elif mapped in {"popsize", "nbest", "nconv", "score_frequency", "flush_frequency", "seed"}:
            cfg[mapped] = int(val)
        elif mapped in {"pmut", "pcross"}:
            cfg[mapped] = float(val)
        else:
            cfg[mapped] = val
        i += 2
    return cfg


def build_args(cfg: dict) -> list[str]:
    args = [str(EXE)]
    args += ["--crossover", cfg["crossover"]]
    args += ["--selector", cfg["selector"]]
    # GAParameterList short names (see gaid.h): popsize, ngen, pmut, pcross, nbest, nconv, sfreq, ffreq, el, seed
    args += [
        "popsize", str(cfg["popsize"]),
        "ngen", str(cfg["generations"]),
        "pmut", str(cfg["pmut"]),
        "pcross", str(cfg["pcross"]),
        "nbest", str(cfg["nbest"]),
        "nconv", str(cfg["nconv"]),
        "sfreq", str(cfg["score_frequency"]),
        "ffreq", str(cfg["flush_frequency"]),
        "el", "1" if cfg.get("elitism", True) else "0",
    ]
    if cfg.get("seed", 0):
        args += ["seed", str(cfg["seed"])]
    return args


def read_stats(path: Path) -> list[dict]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [{k: float(v) for k, v in row.items()} for row in reader]


def moving_plateau(generational_best: list[float], window: int = 20, epsilon: float = 0.001) -> int | None:
    if len(generational_best) < window + 1:
        return None
    for idx in range(window, len(generational_best)):
        past = generational_best[idx - window]
        cur = generational_best[idx]
        if past <= 0:
            continue
        rel_change = abs(past - cur) / past
        if rel_change <= epsilon:
            return idx
    return None


def summarize_run(cfg: dict) -> dict:
    stats_file = ROOT / "cvrp_stats.tsv"
    summary_file = ROOT / "cvrp_summary.txt"
    routes_file = ROOT / "cvrp_best_routes.txt"

    stats_rows = read_stats(stats_file)
    if not stats_rows:
        raise SystemExit("No stats recorded.")
    best_row = min(stats_rows, key=lambda r: r["min"])
    gen0 = stats_rows[0]
    final = stats_rows[-1]
    best_curve = [r["min"] for r in stats_rows]

    plateau = moving_plateau(best_curve, window=25, epsilon=0.001)
    diversity_drop = gen0.get("diversity", -1) - final.get("diversity", -1)
    mean_of_mean = mean(r["mean"] for r in stats_rows)
    dev_of_mean = pstdev(r["mean"] for r in stats_rows)

    summary = {}
    if summary_file.exists():
        with summary_file.open() as f:
            for line in f:
                if not line.strip():
                    continue
                k, v = line.strip().split("\t", 1)
                try:
                    summary[k] = float(v)
                except ValueError:
                    summary[k] = v

    best_cost = summary.get("best_cost", best_row["min"])
    result = {
        "params": cfg,
        "best_cost": best_cost,
        "best_generation": int(best_row["generation"]),
        "initial_best": gen0["min"],
        "final_best": final["min"],
        "improvement_pct": 100.0 * (gen0["min"] - final["min"]) / gen0["min"],
        "plateau_generation": plateau,
        "diversity_drop": diversity_drop,
        "mean_of_mean": mean_of_mean,
        "std_of_mean": dev_of_mean,
        "offlineMax": summary.get("offlineMax"),
        "offlineMin": summary.get("offlineMin"),
        "online": summary.get("online"),
        "convergence": summary.get("convergence"),
        "crossovers": summary.get("crossovers"),
        "mutations": summary.get("mutations"),
        "selections": summary.get("selections"),
        "replacements": summary.get("replacements"),
        "stats_file": str(stats_file),
        "summary_file": str(summary_file),
        "routes_file": str(routes_file),
    }
    return result


def main() -> None:
    if not EXE.exists():
        raise SystemExit(f"Executable not found: {EXE}. Build it with `make cvrp` from examples/.")
    cfg = parse_overrides(sys.argv) if len(sys.argv) > 1 else CONFIG.copy()
    args = build_args(cfg)
    print("Running:", " ".join(args))
    proc = subprocess.run(args, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"cvrp run failed with code {proc.returncode}")
    result = summarize_run(cfg)
    out_json = ROOT / "cvrp_run.json"
    out_json.write_text(json.dumps(result, indent=2))
    # Append a simple log of parameters and best cost for auditing.
    log_path = ROOT / "cvrp_runs.txt"
    with log_path.open("a") as lf:
        lf.write(json.dumps({"params": args, "best_cost": result.get("best_cost")}) + "\n")
    print("Done. Stats are in cvrp_stats.tsv; analyze with analyze_cvrp_ga.py.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
