"""Assistant tools and the OpenAI-compatible tool loop, checked with a mock client."""

import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from assistant import MissingConfiguration, OpenAICompatibleAssistant, offline_answer
from graph_tools import GraphIndex, UnknownGid

ROOT = Path(__file__).resolve().parents[1]


class GraphToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = GraphIndex(ROOT / "data")
        cls.edges = pd.read_parquet(ROOT / "data" / "edges.parquet")
        cls.top = cls.index.top_nodes(limit=5)["shown"]

    def test_profile_and_counterparties_match_data(self):
        gid = int(self.top[0]["gid"])
        profile = self.index.node_profile(gid)
        self.assertEqual(profile["priority_rank"], 1)
        parties = self.index.counterparties(gid)
        expected = ((self.edges.src == gid) | (self.edges.dst == gid)).sum()
        self.assertEqual(parties["total"], expected)
        amounts = [row["sum_kzt"] for row in parties["shown"]]
        self.assertEqual(amounts, sorted(amounts, reverse=True))

    def test_shared_downstream_is_reachable_from_every_listed_source(self):
        gids = [row["gid"] for row in self.top]
        result = self.index.shared_counterparties(gids, "downstream", max_hops=3)
        self.assertGreater(result["total"], 0)
        for row in result["shown"]:
            self.assertGreaterEqual(row["reached_from"], 2)
            self.assertNotIn(row["gid"], gids)
            for source, hops in row["hops_by_source"].items():
                self.assertLessEqual(hops, 3)
                self.assertTrue(self.index.money_paths(source, row["gid"], max_hops=hops)["total"] > 0)

    def test_paths_follow_edges(self):
        edge = next(self.edges.sort_values("sum_kzt", ascending=False).itertuples(index=False))
        result = self.index.money_paths(edge.src, edge.dst, max_hops=1)
        self.assertEqual(result["shown"][0]["amounts_kzt"], [edge.sum_kzt])

    def test_unknown_gid(self):
        with self.assertRaises(UnknownGid):
            self.index.node_profile("123")
        self.assertIn("отсутствует", offline_answer(self.index, "кто платит 123456789012345?"))

    def test_offline_answer_uses_gids(self):
        answer = offline_answer(self.index, f"кто собирает с {self.top[0]['gid']} и {self.top[1]['gid']}?")
        self.assertIn("Общие получатели", answer)


class OpenAICompatibleLoopTest(unittest.TestCase):
    """Replays two API turns: a tool call, then the final answer."""

    def test_tool_loop_runs_graph_tools(self):
        index = GraphIndex(ROOT / "data")
        gid = index.top_nodes(limit=1)["shown"][0]["gid"]
        requests = []

        class Completions:
            def create(self, **request):
                requests.append(copy.deepcopy(request))
                if len(requests) == 1:
                    tool_call = SimpleNamespace(
                        id="call_1", type="function",
                        function=SimpleNamespace(name="node_profile", arguments=json.dumps({"gid": gid})))
                    message = SimpleNamespace(content=None, tool_calls=[tool_call])
                else:
                    message = SimpleNamespace(content=f"Узел {gid}: признаки координации.", tool_calls=None)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        assistant = OpenAICompatibleAssistant(index, "local-model", client=client)
        self.assertEqual(assistant.ask(f"Что за узел {gid}?"), f"Узел {gid}: признаки координации.")

        first, second = requests
        self.assertEqual(first["model"], "local-model")
        self.assertIn("node_profile", {tool["function"]["name"] for tool in first["tools"]})
        tool_result = second["messages"][-1]
        self.assertEqual(tool_result["role"], "tool")
        self.assertEqual(json.loads(tool_result["content"])["gid"], gid)
        # The follow-up question carries the whole previous exchange.
        self.assertEqual(len(assistant.history), 4)

    def test_endpoint_is_required_without_injected_client(self):
        index = GraphIndex(ROOT / "data")
        with self.assertRaisesRegex(MissingConfiguration, "OPENAI_BASE_URL"):
            OpenAICompatibleAssistant(index, "local-model")


if __name__ == "__main__":
    unittest.main()
