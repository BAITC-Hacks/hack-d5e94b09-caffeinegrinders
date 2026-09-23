#!/usr/bin/env python3
"""AML analyst assistant: a natural-language question -> an answer grounded in the graph.

An OpenAI-compatible LLM only plans queries and writes the answer; every fact comes
from deterministic tools in graph_tools.py. Without a configured endpoint the
assistant falls back to the same tools with a rule-based choice of query.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from graph_tools import GraphIndex, UnknownGid

ROOT = Path(__file__).resolve().parent
GID_PATTERN = re.compile(r"\b\d{12,20}\b")

SYSTEM = """Ты — ассистент AML-аналитика банка. Данные: обезличенный граф внутрибанковских \
переводов за июль 2026, собранный от 81 seed-клиента по исходящим переводам на 4 колена, \
порог 5 000 KZT. Сводка выгрузки: {overview}

Правила:
- Любой факт (gid, сумма, роль, связь) бери только из результатов инструментов. Если \
инструменты не дали ответа, так и скажи.
- Указывай gid полностью, как в данных, чтобы аналитик нашёл узел на схеме.
- Формулируй выводы как гипотезы для проверки («признаки консолидации»), не как вину.
- Учитывай ограничения: у узлов 4-го колена исходящие не выгружены (boundary), у seed \
занижены входящие, суммы ниже 5 000 KZT и межбанковские переводы не видны.
- Отвечай по-русски, кратко: вывод, затем узлы с числами, затем что проверить дальше."""


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def function_tool(name: str, description: str, properties: dict, required: list[str] | None = None):
    """Build the standard Chat Completions function-tool schema."""
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required or [], "additionalProperties": False}}}


GID = {"type": "string", "description": "Client identifier, digits only."}
LIMIT = {"type": "integer", "minimum": 1, "maximum": 50}
TOOL_DEFINITIONS = [
    function_tool("node_profile", "Role, priority, metrics, evidence and flags of one client.",
                  {"gid": GID}, ["gid"]),
    function_tool("counterparties", "Direct payers and/or recipients, largest sums first.",
                  {"gid": GID, "direction": {"type": "string", "enum": ["in", "out", "both"]},
                   "limit": LIMIT}, ["gid"]),
    function_tool("shared_counterparties", "Common downstream recipients or upstream sources within N transfers.",
                  {"gids": {"type": "array", "items": GID, "minItems": 1},
                   "direction": {"type": "string", "enum": ["downstream", "upstream"]},
                   "max_hops": {"type": "integer", "minimum": 1, "maximum": 4}, "limit": LIMIT},
                  ["gids"]),
    function_tool("money_paths", "Directed transfer chains between two clients, with amounts per hop.",
                  {"src": GID, "dst": GID,
                   "max_hops": {"type": "integer", "minimum": 1, "maximum": 6}}, ["src", "dst"]),
    function_tool("top_nodes", "Clients in priority order, optionally filtered.",
                  {"role": {"type": "string", "enum": ["coordinator", "consolidator", "distributor",
                                                            "transit", "terminal", "peripheral"]},
                   "cluster_id": {"type": "integer", "minimum": 1},
                   "flagged_only": {"type": "boolean"}, "limit": LIMIT}),
    function_tool("cluster_summary", "Size, seeds, turnover, roles and leaders of one cluster.",
                  {"cluster_id": {"type": "integer", "minimum": 1}}, ["cluster_id"]),
    function_tool("node_cycles", "Directed cycles of up to four transfers through one client.",
                  {"gid": GID, "limit": LIMIT}, ["gid"]),
]


class MissingConfiguration(RuntimeError):
    pass


class OpenAICompatibleAssistant:
    """Tool-calling loop over an explicitly configured OpenAI-compatible endpoint."""

    def __init__(self, index: GraphIndex, model: str, base_url: str | None = None,
                 api_key: str | None = None, client=None):
        if not model:
            raise MissingConfiguration("не задана модель (--model или OPENAI_MODEL)")
        if client is None:
            if not base_url:
                raise MissingConfiguration("не задан OPENAI_BASE_URL; публичный API не используется автоматически")
            from openai import OpenAI
            # Local servers commonly require no authentication, while the SDK requires
            # a nonempty value. Internal gateways can provide their token via OPENAI_API_KEY.
            self.client = OpenAI(base_url=base_url,
                                 api_key=api_key or os.environ.get("OPENAI_API_KEY") or "not-required")
        else:
            self.client = client
        self.model = model
        self.index = index
        self.tools = TOOL_DEFINITIONS
        self.system = SYSTEM.format(overview=dumps(index.overview()))
        self.history: list = []

    @staticmethod
    def _limit(value, default):
        return max(1, min(int(value if value is not None else default), 50))

    def _call_tool(self, name: str, arguments: dict) -> str:
        """Validate model-supplied arguments and execute only known read-only tools."""
        try:
            if name == "node_profile":
                result = self.index.node_profile(arguments["gid"])
            elif name == "counterparties":
                direction = arguments.get("direction", "both")
                if direction not in {"in", "out", "both"}:
                    raise ValueError("direction must be in, out or both")
                result = self.index.counterparties(arguments["gid"], direction,
                                                   self._limit(arguments.get("limit"), 15))
            elif name == "shared_counterparties":
                direction = arguments.get("direction", "downstream")
                if direction not in {"downstream", "upstream"}:
                    raise ValueError("direction must be downstream or upstream")
                hops = max(1, min(int(arguments.get("max_hops", 3)), 4))
                result = self.index.shared_counterparties(arguments["gids"], direction, hops,
                                                          self._limit(arguments.get("limit"), 10))
            elif name == "money_paths":
                hops = max(1, min(int(arguments.get("max_hops", 4)), 6))
                result = self.index.money_paths(arguments["src"], arguments["dst"], hops)
            elif name == "top_nodes":
                result = self.index.top_nodes(arguments.get("role"), arguments.get("cluster_id"),
                                              bool(arguments.get("flagged_only", False)),
                                              self._limit(arguments.get("limit"), 10))
            elif name == "cluster_summary":
                result = self.index.cluster_summary(int(arguments["cluster_id"]))
            elif name == "node_cycles":
                result = self.index.node_cycles(arguments["gid"],
                                                self._limit(arguments.get("limit"), 5))
            else:
                raise ValueError(f"unknown tool: {name}")
            return dumps(result)
        except (KeyError, TypeError, ValueError) as error:
            return dumps({"error": str(error)})

    def ask(self, question: str) -> str:
        messages = [{"role": "system", "content": self.system}, *self.history,
                    {"role": "user", "content": question}]
        for _ in range(12):
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=self.tools, tool_choice="auto")
            if not completion.choices:
                raise RuntimeError("LLM endpoint returned no choices")
            message = completion.choices[0].message
            calls = message.tool_calls or []
            assistant_message = {"role": "assistant", "content": message.content or ""}
            if calls:
                assistant_message["tool_calls"] = [
                    {"id": call.id, "type": "function",
                     "function": {"name": call.function.name,
                                  "arguments": call.function.arguments}}
                    for call in calls
                ]
            messages.append(assistant_message)
            if not calls:
                self.history = messages[1:]
                return message.content.strip() if message.content else "Модель вернула пустой ответ."
            for call in calls:
                try:
                    arguments = json.loads(call.function.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    content = self._call_tool(call.function.name, arguments)
                except (json.JSONDecodeError, ValueError) as error:
                    content = dumps({"error": f"invalid tool arguments: {error}"})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        raise RuntimeError("LLM exceeded the 12-step tool-call limit")


def offline_answer(index: GraphIndex, question: str) -> str:
    """Rule-based fallback: pick one tool from the gids in the question."""
    gids = GID_PATTERN.findall(question)
    try:
        if len(gids) >= 2:
            lowered = question.lower()
            direction = "upstream" if any(w in lowered for w in ("кто платит", "откуда", "источник")) else "downstream"
            result = index.shared_counterparties(gids, direction)
            lines = [f"Общие {'источники' if direction == 'upstream' else 'получатели'} для {len(result['sources'])} gid "
                     f"(до {result['max_hops']} переводов), найдено {result['total']}:"]
            lines += [f"  {r['gid']}  {r['role']}, связан с {r['reached_from']} из {len(result['sources'])}, "
                      f"приоритет #{r['priority_rank']}" for r in result["shown"]]
            return "\n".join(lines)
        if len(gids) == 1:
            profile = index.node_profile(gids[0])
            parties = index.counterparties(gids[0], limit=5)
            lines = [f"{profile['gid']}: {profile['role']}, приоритет #{profile['priority_rank']}, кластер {profile['cluster_id']}",
                     f"  {profile['evidence']}", f"  Обратить внимание: {profile['attention']}"]
            lines += [f"  {'←' if r['direction'] == 'in' else '→'} {r['gid']}  {r['sum_kzt']:,.0f} KZT ({r['role']})"
                      for r in parties["shown"]]
            return "\n".join(lines)
    except UnknownGid as error:
        return str(error)
    result = index.top_nodes(limit=10)
    return "\n".join(["gid в вопросе не найдены. Первые 10 узлов очереди проверки:"]
                     + [f"  #{r['priority_rank']} {r['gid']}  {r['role']}: {r['evidence']}" for r in result["shown"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*", help="вопрос; без него запускается диалог")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"),
                        help="URL локального/внутреннего OpenAI-compatible API (или OPENAI_BASE_URL)")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"),
                        help="имя модели на endpoint (или OPENAI_MODEL)")
    parser.add_argument("--offline", action="store_true", help="без обращения к LLM API")
    args = parser.parse_args()
    index = GraphIndex(args.data)

    assistant = None
    if not args.offline:
        try:
            import openai
        except ImportError:
            print("Пакет openai не установлен; работаю офлайн.", file=sys.stderr)
        else:
            try:
                assistant = OpenAICompatibleAssistant(index, args.model, args.base_url)
            except (MissingConfiguration, openai.OpenAIError, ValueError) as error:
                print(f"LLM API не настроен ({error}); работаю офлайн.", file=sys.stderr)

    def answer(question: str) -> str:
        nonlocal assistant
        if assistant is not None:
            try:
                return assistant.ask(question)
            except (openai.OpenAIError, RuntimeError) as error:
                print(f"LLM API недоступен ({type(error).__name__}); работаю офлайн.", file=sys.stderr)
                assistant = None
        return offline_answer(index, question)

    if args.question:
        print(answer(" ".join(args.question)))
        return
    print("Вопрос по графу (пустая строка — выход):")
    try:
        while question := input("> ").strip():
            print(answer(question), end="\n\n")
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
