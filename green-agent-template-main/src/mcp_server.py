import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO_ROOT / "examples" / "run_cvrp.py"


PROBLEM_DESCRIPTION = {
    "problem_class": "Capacitated vehicle routing problem (CVRP)",
    "representation": {
        "genome": "permutation",
        "dimension": "num_customers",
    },
    "constraints": {
        "method": "penalty",
        "notes": "Capacity constraint handled in fitness with penalties.",
    },
    "objective": {
        "direction": "minimize",
        "metric": "route_cost",
    },
    "fitness_cost_profile": "moderate",
    "instance_distribution": {
        "size_range": "small-to-medium",
        "demand_tightness": "mixed",
    },
    "scoring": {
        "metric": "best_cost",
        "aggregation": "best-of-run",
    },
}


SCHEMA_VERSION = "2026-01-13-mutation-type"
TRIAL_CAP = 5
WALL_CLOCK_LIMIT_SEC = 60

PARAMETERS = {
    "crossover": {
        "type": "enum",
        "values": ["order", "cycle", "onepoint", "twopoint", "partialmatch","uniform", "EvenOdd"],
        "default": "order",
        "aliases": ["crossover_type"],
        "description": "Permutation crossover operator.",
    },
    "selector": {
        "type": "enum",
        "values": ["roulette", "tournament", "rank"],
        "default": "roulette",
        "aliases": ["selector_type"],
        "description": "Selection strategy.",
    },
    "popsize": {
        "type": "int",
        "min": 10,
        "max": 500,
        "default": 100,
        "aliases": ["population_size"],
        "description": "Population size.",
    },
    "generations": {
        "type": "int",
        "min": 10,
        "max": 300,
        "default": 200,
        "aliases": ["ngen"],
        "description": "Number of generations.",
    },
    "pmut": {
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.03,
        "aliases": ["mutation_probability"],
        "description": "Mutation probability.",
    },
    "pcross": {
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.9,
        "aliases": ["crossover_probability"],
        "description": "Crossover probability.",
    },
    "mutation_type": {
        "type": "enum",
        "values": ["single_point", "double_point", "swap", "uniform"],
        "default": "single_point",
        "aliases": ["mutation_operator"],
        "description": "Mutation operator (currently accepted but not applied).",
    },
    "nbest": {
        "type": "int",
        "min": 1,
        "max": 100,
        "default": 5,
        "aliases": [],
        "description": "Number of elites for elitism/replacement.",
    },
    "nconv": {
        "type": "int",
        "min": 1,
        "max": 500,
        "default": 25,
        "aliases": [],
        "description": "Convergence window.",
    },
    "score_frequency": {
        "type": "int",
        "min": 1,
        "max": 200,
        "default": 1,
        "aliases": ["sfreq"],
        "description": "Score sampling frequency.",
    },
    "flush_frequency": {
        "type": "int",
        "min": 1,
        "max": 200,
        "default": 20,
        "aliases": ["ffreq"],
        "description": "Stats flush frequency.",
    },
    "elitism": {
        "type": "bool",
        "default": True,
        "aliases": [],
        "description": "Whether to use elitism.",
    },
    "seed": {
        "type": "int",
        "min": 0,
        "max": 2**31 - 1,
        "default": 0,
        "aliases": [],
        "description": "Random seed (0 means random).",
    },
}


PARAM_EXPLAIN = {
    "crossover": "EvenOdd crossover alternates genes from parents; order preserves relative order; onepoint and twopoint cut and splice; partialmatch ensures valid permutations; uniform randomly selects genes from parents; cycle preserves position-based cycles.",
    "selector": "Balance selection pressure and diversity",
    "popsize": "Larger populations explore more but cost more per generation.",
    "generations": "More generations allow solution refinement but also mean higher computational cost, which is not desirable; recommended using as little generations as possible; trade off with population size.",
    "pmut": "Higher mutation improves exploration but may disrupt structure.",
    "pcross": "Higher crossover emphasizes recombination; too high can overwhelm mutation.",
    "mutation_type": "Mutation operator choice (currently accepted but not applied in the runner).",
    "nbest": "Number of elites retained; higher values reduce diversity.",
    "nconv": "Early stopping window; lower values terminate sooner.",
    "score_frequency": "How often stats are sampled; higher values reduce overhead.",
    "flush_frequency": "How often stats are flushed to disk.",
    "elitism": "Keeps top solutions; useful for stability but can reduce diversity.",
    "seed": "Fixes randomness for reproducibility.",
}

_session_limits: dict[int, dict[str, Any]] = {}


def _session_key(ctx: Context | None) -> int:
    if ctx is None:
        return 0
    return id(ctx.session)


def _get_session_state(ctx: Context | None) -> dict[str, Any]:
    key = _session_key(ctx)
    state = _session_limits.get(key)
    if state is None:
        state = {"count": 0, "start": time.monotonic()}
        _session_limits[key] = state
    return state


def _check_limits(ctx: Context | None) -> dict[str, Any] | None:
    state = _get_session_state(ctx)
    elapsed = time.monotonic() - state["start"]
    remaining_trials = max(TRIAL_CAP - state["count"], 0)
    remaining_seconds = max(WALL_CLOCK_LIMIT_SEC - elapsed, 0.0)
    if elapsed > WALL_CLOCK_LIMIT_SEC:
        return {
            "error": "Trial budget exceeded",
            "details": [f"Wall-clock limit exceeded: {elapsed:.2f}s > {WALL_CLOCK_LIMIT_SEC}s"],
            "remaining_trials": remaining_trials,
            "remaining_seconds": remaining_seconds,
            "response_format": "json_only",
        }
    if state["count"] >= TRIAL_CAP:
        return {
            "error": "Trial budget exceeded",
            "details": [f"Trial cap exceeded: {state['count']} >= {TRIAL_CAP}"],
            "remaining_trials": remaining_trials,
            "remaining_seconds": remaining_seconds,
            "response_format": "json_only",
        }
    return None


def _budget_status(ctx: Context | None) -> dict[str, Any]:
    state = _get_session_state(ctx)
    elapsed = time.monotonic() - state["start"]
    return {
        "trial_cap": TRIAL_CAP,
        "trials_used": state["count"],
        "remaining_trials": max(TRIAL_CAP - state["count"], 0),
        "wall_clock_limit_sec": WALL_CLOCK_LIMIT_SEC,
        "elapsed_seconds": elapsed,
        "remaining_seconds": max(WALL_CLOCK_LIMIT_SEC - elapsed, 0.0),
        "response_format": "json_only",
    }


def _canonicalize_params(params: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    alias_map = {}
    for key, meta in PARAMETERS.items():
        for alias in meta.get("aliases", []):
            alias_map[alias] = key
    for key, value in params.items():
        canonical_key = alias_map.get(key, key)
        canonical[canonical_key] = value
    return canonical


def _validate_params(params: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for key in params:
        if key not in PARAMETERS:
            errors.append(f"Unknown parameter: {key}. Allowed: {sorted(PARAMETERS.keys())}")

    for key, meta in PARAMETERS.items():
        if key not in params:
            continue
        value = params[key]
        ptype = meta["type"]
        if ptype == "enum":
            if value not in meta["values"]:
                errors.append(f"{key} must be one of {meta['values']}")
        elif ptype == "int":
            if not isinstance(value, int):
                errors.append(f"{key} must be int")
                continue
            if "min" in meta and value < meta["min"]:
                errors.append(f"{key} must be >= {meta['min']}")
            if "max" in meta and value > meta["max"]:
                errors.append(f"{key} must be <= {meta['max']}")
        elif ptype == "float":
            if not isinstance(value, (int, float)):
                errors.append(f"{key} must be float")
                continue
            if "min" in meta and float(value) < meta["min"]:
                errors.append(f"{key} must be >= {meta['min']}")
            if "max" in meta and float(value) > meta["max"]:
                errors.append(f"{key} must be <= {meta['max']}")
        elif ptype == "bool":
            if not isinstance(value, bool):
                errors.append(f"{key} must be bool")
    return len(errors) == 0, errors


def _build_args(params: dict[str, Any]) -> list[str]:
    args = [sys.executable, str(RUN_SCRIPT)]
    if "crossover" in params:
        args += ["--crossover", params["crossover"]]
    if "selector" in params:
        args += ["--selector", params["selector"]]

    key_map = {
        "popsize": "popsize",
        "generations": "ngen",
        "pmut": "pmut",
        "pcross": "pcross",
        "nbest": "nbest",
        "nconv": "nconv",
        "score_frequency": "sfreq",
        "flush_frequency": "ffreq",
        "elitism": "el",
        "seed": "seed",
    }
    for key, arg_name in key_map.items():
        if key not in params:
            continue
        value = params[key]
        if key == "elitism":
            value = "true" if value else "false"
        args += [arg_name, str(value)]
    return args


def _load_run_result() -> dict[str, Any]:
    run_json = RUN_SCRIPT.parent / "cvrp_run.json"
    if not run_json.exists():
        return {"error": "cvrp_run.json missing after run"}
    try:
        return json.loads(run_json.read_text())
    except Exception as exc:
        return {"error": f"could not parse cvrp_run.json: {exc}"}


def ga_problem_describe() -> dict[str, Any]:
    return {**PROBLEM_DESCRIPTION, "response_format": "json_only"}


def galib_parameters_schema() -> dict[str, Any]:
    return {
        "parameters": PARAMETERS,
        "notes": "All parameters are optional; unspecified values use defaults.",
        "response_format": "json_only",
    }


def galib_parameters_explain(name: str) -> dict[str, Any]:
    canonical = _canonicalize_params({name: None})
    key = next(iter(canonical.keys()))
    if key not in PARAMETERS:
        return {"error": f"Unknown parameter: {name}", "response_format": "json_only"}
    return {
        "name": key,
        "description": PARAMETERS[key].get("description", ""),
        "behavior": PARAM_EXPLAIN.get(key, ""),
        "schema": PARAMETERS[key],
        "response_format": "json_only",
    }


def galib_run_trial(params: dict[str, Any], seed: int | None = None, ctx: Context | None = None) -> dict[str, Any]:
    if not RUN_SCRIPT.exists():
        return {"error": f"Runner not found: {RUN_SCRIPT}", "response_format": "json_only"}
    limit_error = _check_limits(ctx)
    if limit_error:
        return limit_error
    canonical = _canonicalize_params(params or {})
    if seed is not None:
        canonical["seed"] = seed
    ok, errors = _validate_params(canonical)
    if not ok:
        return {"error": "Invalid parameters", "details": errors, "response_format": "json_only"}

    state = _get_session_state(ctx)
    state["count"] += 1
    args = _build_args(canonical)
    start = time.time()
    proc = subprocess.run(
        args,
        cwd=RUN_SCRIPT.parent,
        capture_output=True,
        text=True,
        timeout=10,
    )
    elapsed = time.time() - start
    if proc.returncode != 0:
        return {
            "error": "run failed",
            "stderr": proc.stderr.strip(),
            "stdout": proc.stdout.strip(),
            "response_format": "json_only",
        }

    result = _load_run_result()
    result["runtime_sec"] = elapsed
    result["response_format"] = "json_only"
    result.update(_budget_status(ctx))
    return result


def benchmark_budget_status(ctx: Context | None = None) -> dict[str, Any]:
    return _budget_status(ctx)


def build_server(host: str, port: int, mount_path: str) -> FastMCP:
    server = FastMCP(
        "ga-benchmark",
        host=host,
        port=port,
        mount_path=mount_path,
    )
    server.tool(name="ga_problem.describe")(ga_problem_describe)
    server.tool(name="galib.parameters.schema")(galib_parameters_schema)
    server.tool(name="galib.parameters.explain")(galib_parameters_explain)
    server.tool(name="galib.run_trial")(galib_run_trial)
    server.tool(name="benchmark.budget.status")(benchmark_budget_status)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCP server for GA benchmark.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=9020, help="Port to bind")
    parser.add_argument("--mount-path", type=str, default="/", help="Mount path for SSE")
    args = parser.parse_args()

    server = build_server(args.host, args.port, args.mount_path)
    print(
        f"[mcp-server] Starting SSE server at http://{args.host}:{args.port}{args.mount_path.rstrip('/')}/sse",
        flush=True,
    )
    print(
        f"[mcp-server] Schema {SCHEMA_VERSION} keys: {sorted(PARAMETERS.keys())}",
        flush=True,
    )
    try:
        server.run(transport="sse", mount_path=args.mount_path)
    except Exception as exc:
        print(f"[mcp-server] Server failed to start: {exc!r}", flush=True)
        raise


if __name__ == "__main__":
    main()
