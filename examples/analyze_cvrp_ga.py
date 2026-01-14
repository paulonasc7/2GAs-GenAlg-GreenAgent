#!/usr/bin/env python3
"""
Analyze the CVRP GAlib run using the generated statistics files.

Expected inputs (produced by examples/cvrp):
- cvrp_stats.tsv: per-generation metrics (mean, max, min, deviation, diversity).
- cvrp_summary.txt: aggregated GAStatistics counters.
- cvrp_best_routes.txt: human-readable best route layout.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
STATS_FILE = ROOT / "cvrp_stats.tsv"
SUMMARY_FILE = ROOT / "cvrp_summary.txt"
ROUTES_FILE = ROOT / "cvrp_best_routes.txt"


def read_stats(path: Path) -> List[Dict[str, float]]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = []
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def read_summary(path: Path) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            key, val = line.strip().split("\t", 1)
            try:
                summary[key] = float(val)
            except ValueError:
                pass
    return summary


def load_routes(path: Path) -> str:
    return path.read_text().strip()


def moving_plateau(generational_best: List[float], window: int = 20, epsilon: float = 0.001) -> int:
    """Return the first generation where improvement over the trailing window is negligible."""
    if len(generational_best) < window + 1:
        return -1
    for idx in range(window, len(generational_best)):
        past = generational_best[idx - window]
        cur = generational_best[idx]
        if past <= 0:
            continue
        rel_change = abs(past - cur) / past
        if rel_change <= epsilon:
            return idx
    return -1


def main() -> None:
    if not STATS_FILE.exists():
        raise SystemExit(f"Missing stats file: {STATS_FILE}")
    stats = read_stats(STATS_FILE)
    summary = read_summary(SUMMARY_FILE) if SUMMARY_FILE.exists() else {}
    routes_text = load_routes(ROUTES_FILE) if ROUTES_FILE.exists() else "No route file found."

    best_row = min(stats, key=lambda r: r["min"])
    worst_row = max(stats, key=lambda r: r["max"])
    gen0 = stats[0]
    final = stats[-1]

    best_curve = [row["min"] for row in stats]
    plateau_gen = moving_plateau(best_curve, window=25, epsilon=0.001)

    mean_of_mean = mean(row["mean"] for row in stats)
    dev_of_mean = pstdev(row["mean"] for row in stats)
    diversity_drop = gen0["diversity"] - final["diversity"]

    best_cost = summary.get("best_cost", best_row["min"])
    total_generations = int(summary.get("generations", final["generation"]))
    crossovers = summary.get("crossovers")
    mutations = summary.get("mutations")

    print("=== CVRP GA Run Analysis ===")
    print(f"Generations: {total_generations}")
    print(f"Best cost: {best_cost:.3f} at generation {int(best_row['generation'])}")
    print(f"Initial best: {gen0['min']:.3f} -> Final best: {final['min']:.3f} "
          f"({100.0 * (gen0['min'] - final['min']) / gen0['min']:.1f}% improvement)")
    print(f"Average mean score across run: {mean_of_mean:.3f} (std {dev_of_mean:.3f})")
    print(f"Best-ever max (worst case): {worst_row['max']:.3f}")
    print(f"Diversity start/end: {gen0['diversity']:.3f} -> {final['diversity']:.3f} "
          f"(drop {diversity_drop:.3f})")

    if plateau_gen >= 0:
        print(f"Plateau detected near generation {plateau_gen} (<=0.1% improvement over prior 25 gens)")
    else:
        print("No strong plateau detected within the analyzed window.")

    if crossovers is not None and mutations is not None:
        print(f"Operators fired: crossovers {int(crossovers)} ({crossovers/total_generations:.1f}/gen), "
              f"mutations {int(mutations)} ({mutations/total_generations:.1f}/gen)")

    if summary:
        print("\n-- Aggregated GAStatistics --")
        for key in ["maxEver", "minEver", "offlineMax", "offlineMin", "online", "convergence"]:
            if key in summary:
                print(f"{key}: {summary[key]:.3f}")

    print("\n-- Best Routes --")
    print(routes_text)

    print("\n-- Quick Pointers --")
    print("* Inspect the diversity drop: large drop + early plateau can signal premature convergence.")
    print("* offlineMin/offlineMax vs online: wide gap suggests only elites improve; tune selection pressure.")
    print("* Crossovers vs mutations per generation give a sanity check against configured rates.")


if __name__ == "__main__":
    main()
