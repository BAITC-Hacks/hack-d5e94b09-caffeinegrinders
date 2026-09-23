#!/usr/bin/env python3
"""AML analyst assistant: a natural-language question -> an answer grounded in the graph.

Claude only plans queries and writes the answer; every fact comes from deterministic
tools in graph_tools.py. Without API access the assistant falls back to the same tools
with a rule-based choice of query.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from graph_tools import GraphIndex, UnknownGid

ROOT = Path(__file__).resolve().parent
MODEL = "claude-opus-5"
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


def make_tools(index: GraphIndex):
    from anthropic import beta_tool

    def safe(call):
        try:
            return dumps(call())
        except UnknownGid as error:
            return dumps({"error": str(error)})

    @beta_tool
    def node_profile(gid: str) -> str:
        """Role, priority rank, metrics, evidence and attention flags of one client.

        Args:
            gid: Client identifier, digits only.
        """
        return safe(lambda: index.node_profile(gid))

    @beta_tool
    def counterparties(gid: str, direction: str = "both", limit: int = 15) -> str:
        """Direct payers ("in") and/or recipients ("out") of a client, largest sums first.

        Args:
            gid: Client identifier.
            direction: "in", "out" or "both".
            limit: Maximum rows to return.
        """
        return safe(lambda: index.counterparties(gid, direction, limit))

    @beta_tool
    def shared_counterparties(gids: list[str], direction: str = "downstream", max_hops: int = 3,
                              limit: int = 10) -> str:
        """Clients connected to several given gids through chains of transfers.

        "downstream" finds who receives money originating from the given gids (who collects it);
        "upstream" finds who sends money that reaches them (common sources). With one gid it lists
        everything reachable. Sorted by how many of the given gids are connected, then hops.

        Args:
            gids: Client identifiers to start from.
            direction: "downstream" or "upstream".
            max_hops: Maximum transfers in a chain, 1-4.
            limit: Maximum rows to return.
        """
        return safe(lambda: index.shared_counterparties(gids, direction, max(1, min(max_hops, 4)), limit))

    @beta_tool
    def money_paths(src: str, dst: str, max_hops: int = 4) -> str:
        """Directed transfer chains from one client to another with the amount on each hop.

        Args:
            src: Sender gid.
            dst: Receiver gid.
            max_hops: Maximum transfers in a chain, 1-6.
        """
        return safe(lambda: index.money_paths(src, dst, max(1, min(max_hops, 6))))

    @beta_tool
    def top_nodes(role: str = "", cluster_id: int = -1, flagged_only: bool = False, limit: int = 10) -> str:
        """Clients in priority order, optionally filtered.

        Args:
            role: "" for any, or coordinator, consolidator, distributor, transit, terminal, peripheral.
            cluster_id: -1 for any cluster.
            flagged_only: Only clients with attention flags (cycles, same-day payers, splitting).
            limit: Maximum rows to return.
        """
        return safe(lambda: index.top_nodes(role or None, None if cluster_id < 0 else cluster_id,
                                            flagged_only, limit))

    @beta_tool
    def cluster_summary(cluster_id: int) -> str:
        """Size, seeds, internal turnover, role mix and top clients of a cluster.

        Args:
            cluster_id: Cluster number from node_profile.
        """
        return safe(lambda: index.cluster_summary(cluster_id))

    @beta_tool
    def node_cycles(gid: str, limit: int = 5) -> str:
        """Return flows (directed cycles of up to 4 transfers) passing through a client.

        Args:
            gid: Client identifier.
            limit: Maximum cycles to return.
        """
        return safe(lambda: index.node_cycles(gid, limit))

    return [node_profile, counterparties, shared_counterparties, money_paths,
            top_nodes, cluster_summary, node_cycles]


class MissingCredentials(RuntimeError):
    pass


class ClaudeAssistant:
    def __init__(self, index: GraphIndex, model: str, client=None):
        import anthropic

        self.client = client or anthropic.Anthropic()
        # Credentials come from ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an `ant auth login`
        # profile (token cache); without any of them the first request would fail.
        if not (self.client.api_key or self.client.auth_token or getattr(self.client, "_token_cache", None)):
            raise MissingCredentials("не заданы ANTHROPIC_API_KEY или профиль ant auth login")
        self.model = model
        self.tools = make_tools(index)
        self.system = SYSTEM.format(overview=dumps(index.overview()))
        self.history: list = []

    def ask(self, question: str) -> str:
        messages = [*self.history, {"role": "user", "content": question}]
        runner = self.client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=16000,
            system=self.system,
            thinking={"type": "adaptive"},
            # On a safety decline the API retries on Anthropic's recommended fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            tools=self.tools,
            messages=messages,
            max_iterations=12,
        )
        final = None
        for message in runner:
            messages.append(message.to_param())
            tool_results = runner.generate_tool_call_response()
            if tool_results is not None:
                messages.append(tool_results)
            final = message
        if final is None or final.stop_reason == "refusal":
            return "Запрос отклонён моделью. Переформулируйте вопрос или используйте --offline."
        self.history = messages
        return "\n".join(block.text for block in final.content if block.type == "text").strip()


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
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--offline", action="store_true", help="без обращения к Claude API")
    args = parser.parse_args()
    index = GraphIndex(args.data)

    assistant = None
    if not args.offline:
        try:
            import anthropic
        except ImportError:
            print("Пакет anthropic не установлен; работаю офлайн.", file=sys.stderr)
        else:
            try:
                assistant = ClaudeAssistant(index, args.model)
            except (MissingCredentials, anthropic.AnthropicError) as error:
                print(f"Claude API недоступен ({error}); работаю офлайн.", file=sys.stderr)

    def answer(question: str) -> str:
        nonlocal assistant
        if assistant is not None:
            import anthropic
            try:
                return assistant.ask(question)
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                    anthropic.APIConnectionError) as error:
                print(f"Claude API недоступен ({type(error).__name__}); работаю офлайн.", file=sys.stderr)
                assistant = None
            except anthropic.APIStatusError as error:
                return f"Ошибка Claude API {error.status_code}: {error.message}"
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
