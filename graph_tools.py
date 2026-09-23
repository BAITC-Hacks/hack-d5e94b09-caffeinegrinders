"""Deterministic graph queries behind the analyst assistant; usable without any LLM."""

from __future__ import annotations

from collections import defaultdict
from itertools import islice
from pathlib import Path

import networkx as nx
import pandas as pd

from run import build_model

NODE_FIELDS = ["role", "role_score", "priority_score", "cluster_id", "depth", "is_seed",
               "in_deg", "out_deg", "in_kzt", "out_kzt", "seed_reach", "boundary",
               "p_hidden_outgoing", "cycles", "sync_payers_max", "priority_base",
               "priority_payers", "priority_incoming", "priority_seed_reach",
               "priority_bridge", "priority_recipients", "priority_rapid",
               "priority_role_factor", "priority_boundary_factor", "priority_seed_factor",
               "evidence", "attention"]


class UnknownGid(ValueError):
    pass


def parse_gid(value) -> int:
    try:
        return int(str(value).strip())
    except ValueError:
        raise UnknownGid(f"gid должен быть числом: {value!r}") from None


class GraphIndex:
    def __init__(self, data: Path):
        self.nodes, self.edges, self.graph, df, self.cycles, self.timeline = build_model(data)
        self.df = df.set_index(df.gid.to_numpy())
        # Rank 1 = highest priority, so answers can say "#3 in the queue".
        order = self.df.sort_values(["priority_score", "gid"], ascending=[False, True]).gid
        self.rank = {int(gid): i + 1 for i, gid in enumerate(order)}

    def _check(self, gid) -> int:
        gid = parse_gid(gid)
        if gid not in self.graph:
            raise UnknownGid(f"gid {gid} отсутствует в выгрузке")
        return gid

    def _brief(self, gid: int) -> dict:
        row = self.df.loc[gid]
        return {"gid": str(gid), "role": row.role, "priority_rank": self.rank[gid],
                "priority_score": round(float(row.priority_score), 3),
                "is_seed": bool(row.is_seed), "cluster_id": int(row.cluster_id)}

    def overview(self) -> dict:
        df = self.df
        return {"nodes": len(df), "edges": len(self.edges), "seeds": int(df.is_seed.sum()),
                "clusters": int(df.cluster_id.nunique()), "cycles_up_to_4": len(self.cycles),
                "boundary_nodes": int(df.boundary.sum()),
                "roles": {k: int(v) for k, v in df.role.value_counts().items()},
                "total_kzt": round(float(self.edges.sum_kzt.sum()), 2)}

    def node_profile(self, gid) -> dict:
        gid = self._check(gid)
        row = self.df.loc[gid]
        profile = {"gid": str(gid), "priority_rank": self.rank[gid]}
        for field in NODE_FIELDS:
            value = row[field]
            profile[field] = value.item() if hasattr(value, "item") else value
        return profile

    def counterparties(self, gid, direction: str = "both", limit: int = 15) -> dict:
        gid = self._check(gid)
        rows = []
        if direction in ("in", "both"):
            rows += [{"direction": "in", "other": src, **data} for src, _, data in self.graph.in_edges(gid, data=True)]
        if direction in ("out", "both"):
            rows += [{"direction": "out", "other": dst, **data} for _, dst, data in self.graph.out_edges(gid, data=True)]
        rows.sort(key=lambda r: (-r["sum_kzt"], r["other"]))
        return {"gid": str(gid), "total": len(rows),
                "shown": [{"direction": r["direction"], "sum_kzt": r["sum_kzt"], "n_tx": r["n_tx"],
                           **self._brief(r["other"])} for r in rows[:limit]]}

    def shared_counterparties(self, gids, direction: str = "downstream", max_hops: int = 3, limit: int = 10) -> dict:
        """Nodes reached by money from several given gids (downstream) or sending money to them (upstream)."""
        sources = list(dict.fromkeys(self._check(g) for g in gids))
        graph = self.graph if direction == "downstream" else self.graph.reverse(copy=False)
        reached = defaultdict(dict)
        for source in sources:
            for node, hops in nx.single_source_shortest_path_length(graph, source, cutoff=max_hops).items():
                if node not in sources:
                    reached[node][str(source)] = hops
        need = 2 if len(sources) > 1 else 1
        hits = [(node, via) for node, via in reached.items() if len(via) >= need]
        hits.sort(key=lambda item: (-len(item[1]), min(item[1].values()), self.rank[item[0]]))
        return {"sources": [str(s) for s in sources], "direction": direction, "max_hops": max_hops,
                "total": len(hits),
                "shown": [{**self._brief(node), "reached_from": len(via), "hops_by_source": via}
                          for node, via in hits[:limit]]}

    def money_paths(self, src, dst, max_hops: int = 4, limit: int = 5) -> dict:
        src, dst = self._check(src), self._check(dst)
        paths = []
        for path in islice(nx.all_simple_paths(self.graph, src, dst, cutoff=max_hops), 500):
            amounts = [self.graph[a][b]["sum_kzt"] for a, b in zip(path, path[1:])]
            paths.append({"path": [str(node) for node in path], "hops": len(path) - 1,
                          "bottleneck_kzt": min(amounts), "amounts_kzt": amounts})
        paths.sort(key=lambda p: (p["hops"], -p["bottleneck_kzt"]))
        return {"src": str(src), "dst": str(dst), "max_hops": max_hops, "total": len(paths), "shown": paths[:limit]}

    def top_nodes(self, role: str | None = None, cluster_id: int | None = None,
                  flagged_only: bool = False, limit: int = 10) -> dict:
        df = self.df
        if role:
            df = df[df.role == role]
        if cluster_id is not None:
            df = df[df.cluster_id == int(cluster_id)]
        if flagged_only:
            df = df[df.attention != "нет"]
        df = df.sort_values(["priority_score", "gid"], ascending=[False, True])
        return {"total": len(df),
                "shown": [{**self._brief(int(r.gid)), "evidence": r.evidence, "attention": r.attention}
                          for r in df.head(limit).itertuples(index=False)]}

    def cluster_summary(self, cluster_id: int) -> dict:
        group = self.df[self.df.cluster_id == int(cluster_id)]
        if group.empty:
            raise UnknownGid(f"кластер {cluster_id} не найден")
        members = set(group.gid)
        internal = self.edges[self.edges.src.isin(members) & self.edges.dst.isin(members)]
        leaders = group.sort_values(["priority_score", "gid"], ascending=[False, True]).head(5)
        return {"cluster_id": int(cluster_id), "n_nodes": len(group), "n_seed": int(group.is_seed.sum()),
                "sum_kzt_internal": round(float(internal.sum_kzt.sum()), 2),
                "roles": {k: int(v) for k, v in group.role.value_counts().items()},
                "top": [self._brief(int(g)) for g in leaders.gid]}

    def node_cycles(self, gid, limit: int = 5) -> dict:
        gid = self._check(gid)
        mask = self.cycles.path.map(lambda p: str(gid) in p.split(" → "))
        found = self.cycles[mask]
        return {"gid": str(gid), "total": len(found),
                "shown": found.head(limit).to_dict(orient="records")}
