import argparse
import json
import os
import re
from typing import Dict, List

import uvicorn
from dotenv import load_dotenv

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
from litellm import completion
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

load_dotenv()

'''SYSTEM_PROMPT = """You suggest genetic algorithm hyperparameters.
Return only JSON with these keys:
- crossover_type (e.g., one_point, two_point, uniform)
- crossover_probability (0-1)
- mutation_type (e.g., bit_flip, swap, gaussian)
- mutation_probability (0-1)
- elitism (true/false)
- population_size (integer)
- generations (integer)
Do not include explanations or extra keys."""'''

'''SYSTEM_PROMPT = """You are an expert in suggesting genetic algorithm hyperparameters.
Return only JSON with the keys and values that are asked for:
Do not include explanations or extra keys."""'''

SYSTEM_PROMPT = """You are an expert in suggesting genetic algorithm hyperparameters.
Return only JSON with the required keys and values. No explanation."""

MCP_URL_RE = re.compile(r"mcp_sse_url\s*:\s*(https?://\S+)", re.IGNORECASE)


def prepare_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="ga_parameter_suggestion",
        name="GA parameter suggestion",
        description="Suggests GA hyperparameters based on feedback.",
        tags=["ga", "suggestion"],
        examples=[],
    )
    return AgentCard(
        name="GASuggester",
        description="Purple agent that proposes GA hyperparameters.",
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


class GASuggesterExecutor(AgentExecutor):
    def __init__(
        self,
        model: str,
        base_url: str | None,
        api_key: str | None,
        extra_headers: dict[str, str] | None,
    ):
        self.ctx_id_to_messages: Dict[str, List[dict]] = {}
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.extra_headers = extra_headers or {}
        self.ctx_id_to_mcp: Dict[str, dict] = {}
        self.debug = os.getenv("GA_SUGGESTER_DEBUG", "0") == "1"

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        context_id = context.context_id

        # Track conversation per context_id
        if context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

        messages = self.ctx_id_to_messages[context_id]
        mcp_url = self._extract_mcp_url(user_input)
        if mcp_url:
            self._debug(f"[ga_suggester] MCP URL detected: {mcp_url}")
            assistant_content = await self._run_mcp_trials(context_id, mcp_url, user_input)
        else:
            self._debug("[ga_suggester] No MCP URL detected in input.")
            assistant_content = None

        if assistant_content is None:
            messages.append({"role": "user", "content": user_input})
            assistant_content = await self._call_model(messages)

        messages.append({"role": "assistant", "content": assistant_content})

        await event_queue.enqueue_event(
            new_agent_text_message(assistant_content, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError

    def _extract_mcp_url(self, text: str) -> str | None:
        match = MCP_URL_RE.search(text or "")
        if match:
            return match.group(1)
        return None

    async def _ensure_mcp_context(
        self,
        context_id: str,
        mcp_url: str,
        messages: List[dict],
    ) -> None:
        cached = self.ctx_id_to_mcp.get(context_id)
        if cached and cached.get("url") == mcp_url:
            return

        data = await self._fetch_mcp_context(mcp_url)
        if not data:
            return
        self._debug(
            "[ga_suggester] MCP context loaded: "
            f"problem={'ok' if data.get('problem') else 'missing'}, "
            f"schema_keys={list((data.get('schema') or {}).get('parameters', {}).keys())}"
        )
        self.ctx_id_to_mcp[context_id] = {"url": mcp_url, "data": data}
        context_blob = json.dumps(data, indent=2)
        messages.insert(
            1,
            {
                "role": "system",
                "content": (
                    "MCP context for this evaluation (problem and parameter schema):\n"
                    f"{context_blob}\n"
                    "Use this context when proposing parameters. Return only JSON."
                ),
            },
        )

    async def _fetch_mcp_context(self, mcp_url: str) -> dict | None:
        try:
            async with sse_client(mcp_url) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    problem = await self._call_tool(session, "ga_problem.describe", {})
                    self._debug("[ga_suggester] MCP tool ga_problem.describe loaded")
                    schema = await self._call_tool(session, "galib.parameters.schema", {})
                    self._debug("[ga_suggester] MCP tool galib.parameters.schema loaded")
                    explains = {}
                    if isinstance(schema, dict):
                        params = schema.get("parameters", {})
                        if isinstance(params, dict):
                            for name in params.keys():
                                explains[name] = await self._call_tool(
                                    session, "galib.parameters.explain", {"name": name}
                                )
                    if explains:
                        self._debug("[ga_suggester] MCP tool galib.parameters.explain loaded")
                    return {
                        "problem": problem,
                        "schema": schema,
                        "explain": explains,
                    }
        except Exception:
            return None

    async def _call_tool(self, session: ClientSession, name: str, arguments: dict) -> dict:
        result = await session.call_tool(name, arguments)
        if result.isError:
            return {"error": "tool_error", "name": name}
        if result.structuredContent is not None:
            return result.structuredContent
        for block in result.content:
            if getattr(block, "type", None) == "text":
                try:
                    return json.loads(block.text)
                except Exception:
                    return {"text": block.text}
        return {}

    async def _call_model(self, messages: List[dict]) -> str:
        kwargs = {"model": self.model, "messages": messages}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        try:
            response = completion(**kwargs, temperature=0.8)
            return response.choices[0].message.content
        except Exception as exc:
            return '{"error": "model call failed", "detail": "%s"}' % exc

    async def _run_mcp_trials(self, context_id: str, mcp_url: str, user_input: str) -> str | None:
        messages = self.ctx_id_to_messages[context_id]
        await self._ensure_mcp_context(context_id, mcp_url, messages)

        trial_cap = int(os.getenv("GA_SUGGESTER_TRIALS", "3"))
        history: list[dict] = []
        best_params: dict | None = None
        best_score: float | None = None
        self._debug(f"[ga_suggester] Starting MCP trials: {trial_cap}")

        try:
            self._debug(f"[ga_suggester] Connecting to MCP SSE: {mcp_url}")
            async with sse_client(mcp_url, timeout=5, sse_read_timeout=15) as streams:
                self._debug("[ga_suggester] MCP SSE connected")
                async with ClientSession(*streams) as session:
                    self._debug("[ga_suggester] Initializing MCP session")
                    await session.initialize()
                    self._debug("[ga_suggester] MCP session initialized")
                    for i in range(trial_cap):
                        prompt = self._build_trial_prompt(user_input, history, i + 1, trial_cap)
                        messages.append({"role": "user", "content": prompt})
                        raw = await self._call_model(messages)
                        messages.append({"role": "assistant", "content": raw})

                        params = self._parse_params(raw)
                        if isinstance(params, dict):
                            params = self._normalize_params(params)
                        if not isinstance(params, dict) or "raw_response" in params:
                            history.append({"iteration": i + 1, "params": params, "result": {"error": "bad_params"}})
                            self._debug(f"[ga_suggester] Trial {i + 1}: invalid params")
                            break

                        self._debug(f"[ga_suggester] Trial {i + 1}: calling galib.run_trial with params={params}")
                        result = await self._call_tool(session, "galib.run_trial", {"params": params})
                        if isinstance(result, dict) and result.get("error") == "Invalid parameters":
                            details = result.get("details")
                            if isinstance(details, list):
                                details.append(
                                    "Use only the exact enum values shown in the schema; no synonyms or reformatting."
                                )
                        self._debug(f"[ga_suggester] Trial {i + 1}: result={result}")
                        score = None
                        if isinstance(result, dict) and "best_cost" in result:
                            score = -float(result["best_cost"])
                            if best_score is None or score > best_score:
                                best_score = score
                                best_params = params

                        history.append(
                            {"iteration": i + 1, "params": params, "result": result, "score": score}
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Trial result:\n{json.dumps(result, indent=2)}",
                            }
                        )
        except Exception as exc:
            self._debug(f"[ga_suggester] MCP trials failed: {exc!r}")
            if isinstance(exc, ExceptionGroup):
                for idx, sub in enumerate(exc.exceptions, start=1):
                    self._debug(f"[ga_suggester] MCP sub-error {idx}: {sub!r}")
            return None

        if best_params is not None:
            self._debug(f"[ga_suggester] Returning best params: {best_params}")
            return json.dumps(best_params, indent=2)
        if history:
            last = history[-1].get("params")
            if isinstance(last, dict):
                self._debug(f"[ga_suggester] Returning last params: {last}")
                return json.dumps(last, indent=2)
        return None

    def _build_trial_prompt(
        self,
        user_input: str,
        history: list[dict],
        iteration: int,
        trial_cap: int,
    ) -> str:
        if not history:
            return (
                f"{user_input}\n\n"
                f"Trial {iteration}/{trial_cap}. Propose GA parameters as JSON only."
            )
        last = history[-1]
        return (
            f"Trial {iteration}/{trial_cap}. Previous params and result:\n"
            f"params: {json.dumps(last.get('params'), indent=2)}\n"
            f"result: {json.dumps(last.get('result'), indent=2)}\n"
            "Return improved parameters as JSON only."
        )

    def _parse_params(self, raw: str) -> dict:
        text = (raw or "").strip()
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

    def _normalize_params(self, params: dict) -> dict:
        normalized = dict(params)
        crossover_map = {
            "one_point": "onepoint",
            "one-point": "onepoint",
            "onepoint": "onepoint",
            "two_point": "twopoint",
            "two-point": "twopoint",
            "twopoint": "twopoint",
            "even-odd": "EvenOdd",
            "evenodd": "EvenOdd",
            "partial-match": "partialmatch",
            "partialmatch": "partialmatch",
        }
        selector_map = {
            "roulette_wheel": "roulette",
            "rank_selector": "rank",
            "tournament_selector": "tournament",
        }
        mutation_map = {
            "single-point": "single_point",
            "single_point": "single_point",
            "double-point": "double_point",
            "double_point": "double_point",
            "double-point-mutation": "double_point",
            "swap_mutation": "swap",
        }
        crossover = normalized.get("crossover_type") or normalized.get("crossover")
        if isinstance(crossover, str) and crossover in crossover_map:
            normalized["crossover_type"] = crossover_map[crossover]

        selector = normalized.get("selector_type") or normalized.get("selector")
        if isinstance(selector, str) and selector in selector_map:
            normalized["selector_type"] = selector_map[selector]

        mutation = normalized.get("mutation_type")
        if isinstance(mutation, str) and mutation in mutation_map:
            normalized["mutation_type"] = mutation_map[mutation]

        return normalized

    def _debug(self, message: str) -> None:
        if self.debug:
            print(message, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run the GA purple agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9018, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("GA_SUGGESTER_MODEL", "ollama/llama3.2:latest"),
        help="Model identifier (litellm-compatible)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("LLM_BASE_URL") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Base URL for the LLM provider",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("LLM_API_KEY"),
        help="API key for the LLM provider",
    )
    parser.add_argument(
        "--extra-header",
        action="append",
        default=[],
        help="Extra HTTP header for the LLM provider (KEY=VALUE). Can be used multiple times.",
    )
    args = parser.parse_args()

    card = prepare_agent_card(args.card_url or f"http://{args.host}:{args.port}/")
    extra_headers: dict[str, str] = {}
    for item in args.extra_header:
        if "=" in item:
            key, value = item.split("=", 1)
            extra_headers[key] = value
    executor = GASuggesterExecutor(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        extra_headers=extra_headers or None,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
