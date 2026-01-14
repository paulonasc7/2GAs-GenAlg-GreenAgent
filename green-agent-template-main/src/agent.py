import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


class Agent:
    required_roles: list[str] = ["ga_suggester"]
    required_config_keys: list[str] = ["iterations"]

    def __init__(self):
        self.messenger = Messenger()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        try:
            iterations = int(request.config["iterations"])
            if iterations < 1:
                return False, "iterations must be >= 1"
        except Exception as exc:
            return False, f"Can't parse iterations: {exc}"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        iterations = int(request.config["iterations"])
        suggester_url = str(request.participants["ga_suggester"])
        history: list[dict[str, Any]] = []
        conversation: list[dict[str, Any]] = []
        best_score: float | None = None
        best_params: dict[str, Any] | None = None

        repo_root = Path(__file__).resolve().parents[2]
        run_script = repo_root / "examples" / "run_cvrp.py"
        mcp_script = repo_root / "green-agent-template-main" / "src" / "mcp_server.py"

        mcp_host = str(request.config.get("mcp_host", "127.0.0.1"))
        mcp_port = int(request.config.get("mcp_port", 9020))
        mcp_mount_path = str(request.config.get("mcp_mount_path", "/"))
        mcp_sse_url = self._build_mcp_sse_url(mcp_host, mcp_port, mcp_mount_path)
        mcp_proc = self._start_mcp_server(mcp_script, mcp_host, mcp_port, mcp_mount_path)

        await updater.update_status(
            TaskState.working, new_agent_text_message("Starting GA tuning...")
        )

        try:
            for i in range(iterations):
                prompt = self._build_prompt(history, mcp_sse_url)
                response = await self.messenger.talk_to_agent(
                    prompt, suggester_url, new_conversation=(i == 0)
                )
                params = self._parse_params(response)
                result = self._score_params(params, run_script)

                score = None
                if result and "best_cost" in result:
                    score = float(result["best_cost"])
                    if best_score is None or score < best_score:
                        best_score = score
                        best_params = params

                history.append(
                    {"iteration": i + 1, "params": params, "result": result, "score": score}
                )
                conversation.append(
                    {
                        "iteration": i + 1,
                        "prompt": prompt,
                        "response": response,
                        "parsed_params": params,
                        "result": result,
                    }
                )

                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"Iteration {i + 1}: score={score}\nParams:\n{json.dumps(params, indent=2)}"
                    ),
                )
        finally:
            self._stop_mcp_server(mcp_proc)

        summary = self._build_summary(best_score, best_params, history)
        payload = {
            "best_score": best_score,
            "best_params": best_params,
            "history": history,
            "conversation": conversation,
        }
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary)),
                Part(root=DataPart(data=payload)),
            ],
            name="GA Tuning Result",
        )
        self._write_run_log(payload)

    def _build_prompt(self, history: list[dict[str, Any]], mcp_sse_url: str | None) -> str:
        mcp_hint = ""
        if mcp_sse_url:
            mcp_hint = (
                "Use MCP tools to fetch the problem description and parameter schema.\n"
                f"MCP_SSE_URL: {mcp_sse_url}\n"
            )
        if not history:
            return (
                f"{mcp_hint}"
                "You tune GA parameters for CVRP. The objective is to MINIMIZE best_cost; lower is better."
                "You can find more information about the problem in the MCP tools. "
                "Return your parameter proposal as a JSON object with keys: "
                "crossover_type, selector_type,"
                "crossover_probability, mutation_type, mutation_probability, "
                "elitism (true/false), population_size, generations. "
                "Use concise values only. Occasionally vary categorical operators (crossover_type/selector_type) unless it clearly harms performance. "
                "Return JSON only: no markdown, no code fences, no explanations."
            )
        last = history[-1]
        return (
            f"{mcp_hint}"
            "You suggested parameters earlier. Here is the last result:\n"
            f"result: {json.dumps(last.get('result'), indent=2)}\n"
            f"params: {json.dumps(last.get('params'), indent=2)}\n"
            "Analyze the result above to guide your next choice. "
            "Return an improved JSON payload with the same keys, targeting a LOWER best_cost. "
            "Change at least one parameter unless you believe the previous set is best. "
            "If you repeat the previous set, do so intentionally. "
            "If results are flat, explore a different crossover_type or selector_type. "
            "Return JSON only: no markdown, no code fences, no explanations."
        )

    def _parse_params(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        fence = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"raw_response": raw}

    def _score_params(self, params: dict[str, Any], run_script: Path) -> dict[str, Any]:
        overrides = self._map_params(params)
        arglist = []
        for k, v in overrides.items():
            arglist.extend([k, str(v)])

        cmd = [sys.executable, str(run_script)] + arglist
        try:
            proc = subprocess.run(
                cmd,
                cwd=run_script.parent.parent,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode != 0:
                return {"error": f"run failed: {proc.stderr.strip()}"}
        except Exception as exc:
            return {"error": f"exception during run: {exc}"}

        run_json = run_script.parent / "cvrp_run.json"
        if not run_json.exists():
            return {"error": "cvrp_run.json missing after run"}
        try:
            return json.loads(run_json.read_text())
        except Exception as exc:
            return {"error": f"could not parse cvrp_run.json: {exc}"}

    def _map_params(self, params: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        crossover = params.get("crossover") or params.get("crossover_type")
        if crossover:
            if crossover in ["one_point", "onepoint", "one-point"]:
                out["crossover"] = "onepoint"
            elif crossover in ["two_point", "twopoint", "two-point"]:
                out["crossover"] = "twopoint"
            elif crossover in ["order", "order_crossover", "order-based"]:
                out["crossover"] = "order"
            elif crossover in ["EvenOdd", "evenodd", "even-odd"]:
                out["crossover"] = "EvenOdd"
            elif crossover in ["partialmatch", "partial-match", "PartialMatch"]:
                out["crossover"] = "partialmatch"
            elif crossover in ["cycle", "Cycle"]:
                out["crossover"] = "cycle"
            elif crossover in ["uniform", "Uniform"]:
                out["crossover"] = "uniform"

        selector = params.get("selector") or params.get("selector_type")
        if selector:
            if selector in ["tournament", "tournament_selector"]:
                out["selector"] = "tournament"
            elif selector in ["rank", "rank_selector"]:
                out["selector"] = "rank"
            elif selector in ["roulette", "roulette_wheel"]:
                out["selector"] = "roulette"

        mapping = {
            "popsize": ["population_size", "popsize"],
            "generations": ["generations"],
            "pmut": ["mutation_probability", "pmut"],
            "pcross": ["crossover_probability", "pcross"],
            "nbest": ["nbest"],
            "nconv": ["nconv"],
            "score_frequency": ["score_frequency"],
            "flush_frequency": ["flush_frequency"],
            "seed": ["seed"],
        }
        for target, keys in mapping.items():
            for k in keys:
                if k in params:
                    out[target] = params[k]
                    break
        if "elitism" in params:
            out["elitism"] = params["elitism"]
        return out

    def _build_summary(
        self,
        best_score: float | None,
        best_params: dict[str, Any] | None,
        history: list[dict[str, Any]],
    ) -> str:
        lines = [
            "GA tuning summary",
            f"Best score: {best_score} (lower is better, best_cost)",
        ]
        if best_params is not None:
            lines.append(f"Best params:\n{json.dumps(best_params, indent=2)}")
        lines.append("History:")
        for h in history:
            lines.append(f"- Iteration {h['iteration']}: score={h.get('score')}")
        return "\n".join(lines)

    def _write_run_log(self, payload: dict[str, Any]) -> None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path("ga_runs")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"run-{ts}.json"
        out_path.write_text(json.dumps(payload, indent=2))

    def _build_mcp_sse_url(self, host: str, port: int, mount_path: str) -> str:
        base = f"http://{host}:{port}"
        mount = (mount_path or "/").strip("/")
        if mount:
            base = f"{base}/{mount}"
        return f"{base}/sse"

    def _start_mcp_server(
        self,
        script_path: Path,
        host: str,
        port: int,
        mount_path: str,
    ) -> subprocess.Popen | None:
        if not script_path.exists():
            print(f"[green-agent] MCP server script missing: {script_path}", flush=True)
            return None
        cmd = [
            sys.executable,
            str(script_path),
            "--host",
            host,
            "--port",
            str(port),
            "--mount-path",
            mount_path,
        ]
        print(f"[green-agent] Starting MCP server: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            text=True,
        )
        time.sleep(0.5)
        if proc.poll() is not None:
            print(f"[green-agent] MCP server exited early with code {proc.returncode}", flush=True)
        return proc

    def _stop_mcp_server(self, proc: subprocess.Popen | None) -> None:
        if not proc:
            return
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
