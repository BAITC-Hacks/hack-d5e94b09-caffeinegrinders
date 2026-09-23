#!/usr/bin/env python3
"""One-command, deterministic AML graph analysis and self-contained viewer."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROLES = {"coordinator", "consolidator", "transit", "distributor", "terminal", "peripheral"}
ROOT = Path(__file__).resolve().parent
CYCLE_MAX_LEN = 4
NEAR_THRESHOLD_KZT = 10_000
SYNC_PAYERS_MIN = 3
REPEAT_AMOUNT_MIN = 4
REMOVAL_STEPS = (0, 5, 10, 20, 50, 100)


def load_data(path: Path):
    nodes = pd.read_parquet(path / "nodes.parquet")
    edges = pd.read_parquet(path / "edges.parquet")
    tx = pd.read_parquet(path / "transactions.parquet")
    for frame, columns in ((nodes, {"gid", "depth", "is_seed"}),
                           (edges, {"src", "dst", "sum_kzt", "n_tx"}),
                           (tx, {"src", "dst", "date", "sum_kzt"})):
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
    if nodes.gid.isna().any() or nodes.gid.duplicated().any():
        raise ValueError("nodes.gid must be unique and nonempty")
    if edges.duplicated(["src", "dst"]).any():
        raise ValueError("Duplicate aggregated edges")
    gids = set(nodes.gid)
    if not set(edges.src).union(edges.dst).issubset(gids):
        raise ValueError("Edge endpoint absent from nodes")
    grouped = tx.groupby(["src", "dst"], as_index=False).agg(amount=("sum_kzt", "sum"), count=("sum_kzt", "size"))
    check = edges.merge(grouped, on=["src", "dst"], how="outer", indicator=True)
    if (check._merge != "both").any() or not np.allclose(check.sum_kzt, check.amount) or not (check.n_tx == check["count"]).all():
        raise ValueError("transactions and edges disagree")
    tx["date"] = pd.to_datetime(tx["date"], errors="raise")
    return nodes.sort_values("gid").reset_index(drop=True), edges, tx


def graph_features(nodes: pd.DataFrame, edges: pd.DataFrame, tx: pd.DataFrame):
    graph = nx.DiGraph()
    graph.add_nodes_from(int(x) for x in nodes.gid)
    for row in edges.itertuples(index=False):
        graph.add_edge(int(row.src), int(row.dst), sum_kzt=float(row.sum_kzt), n_tx=int(row.n_tx))

    df = nodes.copy()
    mappings = {
        "in_deg": dict(graph.in_degree()), "out_deg": dict(graph.out_degree()),
        "in_kzt": dict(graph.in_degree(weight="sum_kzt")),
        "out_kzt": dict(graph.out_degree(weight="sum_kzt")),
        "in_tx": dict(graph.in_degree(weight="n_tx")),
        "out_tx": dict(graph.out_degree(weight="n_tx")),
    }
    for key, values in mappings.items():
        df[key] = df.gid.map(values).fillna(0)
        if key.endswith("deg") or key.endswith("tx"):
            df[key] = df[key].astype(int)
        else:
            # Tiyn precision; float summation order must not leak into the CSV.
            df[key] = df[key].round(2)
    df["ratio"] = np.where(df.in_kzt > 0, df.out_kzt / df.in_kzt.replace(0, np.nan), np.nan)
    df["boundary"] = (df.depth == 4) & (df.out_deg == 0)

    # Reachability follows the direction of transfers. The 81 seed traversals are small
    # on this graph and make shared upstream sources explicit.
    seed_reach = Counter()
    for seed in nodes.loc[nodes.is_seed, "gid"]:
        seen = nx.single_source_shortest_path_length(graph, int(seed), cutoff=4)
        seed_reach.update(seen.keys())
    df["seed_reach"] = df.gid.map(seed_reach).fillna(0).astype(int)

    # Sampled shortest-path mediation gives a stable, affordable bridge signal.
    between = nx.betweenness_centrality(graph, k=min(96, len(graph)), normalized=True, seed=42)
    # Rounded so that float noise across platforms does not change the CSV.
    df["bridge"] = df.gid.map(between).fillna(0.0).round(10)
    # Each incoming transfer counts as rapidly relayed if this account sends at
    # least one transfer in the following 48 hours. This is a timing clue, not
    # transaction matching or proof that the same money moved on.
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for row in tx.itertuples(index=False):
        incoming[int(row.dst)].append(row.date)
        outgoing[int(row.src)].append(row.date)
    rapid = {}
    for gid, dates in incoming.items():
        outs = sorted(outgoing.get(gid, []))
        if not outs:
            rapid[gid] = 0.0
            continue
        count = sum(any(0 <= (out - day).total_seconds() <= 172800 for out in outs) for day in dates)
        rapid[gid] = count / len(dates)
    df["rapid_48h"] = df.gid.map(rapid).fillna(0.0)

    # Several distinct payers on the same day can indicate a coordinated collection.
    sync = tx.groupby(["dst", "date"]).src.nunique().groupby("dst").max()
    df["sync_payers_max"] = df.gid.map(sync).fillna(0).astype(int)
    # The same amount sent repeatedly is a simple splitting clue.
    repeats = tx.groupby(["src", "sum_kzt"]).size().groupby("src").max()
    df["repeat_amount_max"] = df.gid.map(repeats).fillna(0).astype(int)
    # Transfers right above the 5 000 KZT cut-off hint at splitting below it.
    near = tx[tx.sum_kzt < NEAR_THRESHOLD_KZT].groupby("dst").size()
    df["near_threshold_in"] = df.gid.map(near).fillna(0).astype(int)

    # Traversal expands every node up to depth 3, so their outgoing transfers are
    # visible. For depth-4 nodes without outgoing we estimate how often comparable
    # nodes at depths 1-3 (same payer-count bucket) do send money further.
    expanded = df[df.depth.between(1, 3) & ~df.is_seed & (df.in_deg > 0)]
    continuation = expanded.groupby(payer_bucket(expanded.in_deg)).out_deg.apply(lambda s: float((s > 0).mean()))
    expected = payer_bucket(df.in_deg).astype(object).map(continuation).astype(float).fillna(0.0)
    df["p_hidden_outgoing"] = np.where(df.boundary, expected, 0.0).round(3)
    return graph, df


def payer_bucket(in_deg: pd.Series):
    return pd.cut(in_deg, [0, 1, 2, 4, math.inf], labels=["1", "2", "3-4", "5+"])


def find_cycles(graph: nx.DiGraph, df: pd.DataFrame):
    """Return flows: simple directed cycles of up to four transfers."""
    seeds = set(df.gid[df.is_seed])
    rows, per_node, seed_cycle = [], Counter(), set()
    for path in nx.simple_cycles(graph, length_bound=CYCLE_MAX_LEN):
        # Rotate so that equal cycles always print from the same node.
        start = path.index(min(path))
        path = path[start:] + path[:start]
        hops = list(zip(path, path[1:] + path[:1]))
        n_seed = sum(node in seeds for node in path)
        rows.append({"length": len(path),
                     "path": " → ".join(str(node) for node in path + path[:1]),
                     "bottleneck_kzt": round(min(graph[a][b]["sum_kzt"] for a, b in hops), 2),
                     "n_seed": n_seed})
        per_node.update(path)
        if n_seed:
            seed_cycle.update(path)
    df["cycles"] = df.gid.map(per_node).fillna(0).astype(int)
    df["cycle_with_seed"] = df.gid.isin(seed_cycle)
    cycles = pd.DataFrame(rows, columns=["length", "path", "bottleneck_kzt", "n_seed"])
    cycles = cycles.sort_values(["bottleneck_kzt", "length", "path"], ascending=[False, True, True]).reset_index(drop=True)
    cycles.insert(0, "cycle_id", range(1, len(cycles) + 1))
    return df, cycles


def flag_attention(df: pd.DataFrame):
    """Short, checkable hints for the analyst; they do not change role or priority."""
    depth_cut = df.groupby("depth").in_deg.transform(lambda s: s.quantile(.99))
    notes = []
    for r, cut in zip(df.itertuples(index=False), depth_cut):
        items = []
        if r.cycle_with_seed:
            items.append(f"возвратный поток с seed ({r.cycles} цикл.)")
        elif r.cycles:
            items.append(f"возвратный поток ({r.cycles} цикл.)")
        if r.sync_payers_max >= SYNC_PAYERS_MIN:
            items.append(f"{r.sync_payers_max} плательщиков в один день")
        if r.repeat_amount_max >= REPEAT_AMOUNT_MIN:
            items.append(f"повтор одной суммы ×{r.repeat_amount_max}")
        if r.near_threshold_in >= 3 and r.near_threshold_in >= .5 * r.in_tx:
            items.append(f"{r.near_threshold_in} входов у порога 5–10 тыс.")
        if r.in_deg >= 3 and r.in_deg > cut:
            items.append(f"плательщиков больше 99% узлов колена {r.depth}")
        if r.boundary and r.p_hidden_outgoing >= .5:
            items.append(f"продлить обход: скрытый выход ~{r.p_hidden_outgoing:.0%}")
        notes.append("; ".join(items)[:200] if items else "нет")
    df["attention"] = notes
    return df


def assign_clusters(graph: nx.DiGraph, df: pd.DataFrame, edges: pd.DataFrame):
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for src, dst, amount in edges[["src", "dst", "sum_kzt"]].itertuples(index=False, name=None):
        if undirected.has_edge(src, dst):
            undirected[src][dst]["weight"] += float(amount)
        else:
            undirected.add_edge(src, dst, weight=float(amount))
    communities = nx.community.louvain_communities(undirected, weight="weight", resolution=1, seed=42)
    communities.sort(key=lambda group: (-len(group), min(group)))
    ids = {int(gid): index + 1 for index, group in enumerate(communities) for gid in group}
    df["cluster_id"] = df.gid.map(ids).astype(int)
    return df


def score_roles(df: pd.DataFrame):
    # A high bridge threshold is relative to this batch; all other thresholds
    # are fixed and listed in README.
    bridge_cut = float(df.loc[df.bridge > 0, "bridge"].quantile(.90)) if (df.bridge > 0).any() else math.inf
    roles, confidences, evidence = [], [], []
    for r in df.itertuples(index=False):
        ratio = r.ratio
        if r.seed_reach >= 3 and r.in_deg >= 3 and r.bridge >= bridge_cut and not r.is_seed:
            role = "coordinator"
            confidence = min(.95, .60 + .05 * min(r.seed_reach - 3, 4) + .06 * min(r.in_deg - 3, 3))
            reason = f"Гипотеза координации: достижим из {r.seed_reach} seed, {r.in_deg} плательщиков, мост {r.bridge:.4f}."
        elif r.out_deg >= 8 and r.out_deg >= 2 * max(r.in_deg, 1):
            role = "distributor"
            confidence = min(.95, .60 + .025 * min(r.out_deg - 8, 10) + .03 * min(r.out_deg / max(r.in_deg, 1) - 2, 3))
            reason = f"Признаки распределения: {r.out_deg} получателей против {r.in_deg} плательщиков, исходящий поток {r.out_kzt:,.0f} KZT."
        elif r.in_deg >= 4 and r.in_deg >= 2 * max(r.out_deg, 1):
            role = "consolidator"
            confidence = min(.95, .60 + .045 * min(r.in_deg - 4, 6) + .025 * min(r.in_deg / max(r.out_deg, 1) - 2, 4))
            reason = f"Признаки консолидации: {r.in_deg} плательщиков, {r.out_deg} получателей, входящий поток {r.in_kzt:,.0f} KZT."
        elif not r.is_seed and r.in_deg > 0 and r.out_deg > 0 and pd.notna(ratio) and .8 <= ratio <= 1.2:
            role = "transit"
            confidence = min(.90, .60 + .15 * (1 - abs(ratio - 1) / .2) + .10 * r.rapid_48h)
            reason = f"Признаки транзита: отдал {ratio:.0%} видимого входа; {r.in_deg}→{r.out_deg} связей, быстрый выход {r.rapid_48h:.0%}."
        elif r.in_deg > 0 and r.out_deg == 0 and r.depth < 4 and not r.is_seed:
            role = "terminal"
            confidence = .62 + .04 * min(r.in_deg - 1, 5)
            reason = f"Наблюдаемый конечный узел: вход от {r.in_deg} клиентов ({r.in_kzt:,.0f} KZT), исходящих нет; колено {r.depth}<4."
        else:
            role = "peripheral"
            confidence = .55 if r.boundary else .60
            if r.boundary:
                reason = (f"Граница обхода: колено 4, вход от {r.in_deg} клиентов; исходящие не выгружены. "
                          f"У похожих узлов колен 1–3 выход есть в {r.p_hidden_outgoing:.0%} случаев.")
            elif r.is_seed and r.out_deg == 0:
                reason = "Seed без исходящих в выборке; входящий поток неполон, структурная роль не установлена."
            else:
                reason = f"Недостаточно признаков роли: {r.in_deg} плательщиков, {r.out_deg} получателей, колено {r.depth}."
        roles.append(role)
        confidences.append(round(float(confidence), 4))
        evidence.append(reason[:200])
    df["role"] = roles
    df["role_score"] = confidences
    df["evidence"] = evidence
    return df


def rank_nodes(df: pd.DataFrame):
    # Percentile ranks keep unlike units comparable; ties receive the same rank.
    signals = {
        "payers": df.in_deg.rank(pct=True),
        "incoming": df.in_kzt.rank(pct=True),
        "seeds": df.seed_reach.rank(pct=True),
        "bridge": df.bridge.rank(pct=True),
        "recipients": df.out_deg.rank(pct=True),
    }
    score = (.20 * signals["payers"] + .20 * signals["incoming"]
             + .20 * signals["seeds"] + .20 * signals["bridge"]
             + .15 * signals["recipients"] + .05 * df.rapid_48h)
    score *= df.role.map({"coordinator": 1.0, "consolidator": 1.0,
                          "distributor": 1.0, "transit": .95,
                          "terminal": .85, "peripheral": .75})
    score *= np.where(df.boundary, .65, 1.0)
    score *= np.where(df.is_seed, .85, 1.0)
    df["priority_score"] = score.clip(0, 1).round(6)
    return df


def seed_reach_pairs(graph: nx.DiGraph, seeds: list[int]):
    """Number of (seed, node) pairs connected by at most four transfers."""
    return sum(len(nx.single_source_shortest_path_length(graph, seed, cutoff=4)) - 1
               for seed in seeds if seed in graph)


def network_resilience(graph: nx.DiGraph, df: pd.DataFrame):
    """What remains of the network after removing the top-N nodes versus N random ones."""
    seeds = [int(x) for x in df.gid[df.is_seed]]
    by_priority = df.sort_values(["priority_score", "gid"], ascending=[False, True]).gid.astype(int).tolist()
    at_random = [int(x) for x in np.random.default_rng(42).permutation(df.gid.to_numpy())]
    base_reach = seed_reach_pairs(graph, seeds)
    rows = []
    for strategy, order in (("priority", by_priority), ("random", at_random)):
        for n in REMOVAL_STEPS:
            rest = graph.copy()
            rest.remove_nodes_from(order[:n])
            components = [len(c) for c in nx.weakly_connected_components(rest) if len(c) > 1]
            reach = seed_reach_pairs(rest, seeds)
            rows.append({"strategy": strategy, "removed": n,
                         "largest_component": max(components, default=0),
                         "linked_components": len(components),
                         "edges_left": rest.number_of_edges(),
                         "seed_reach_pairs": reach,
                         "seed_reach_share": round(reach / base_reach, 4) if base_reach else 0.0})
    return pd.DataFrame(rows)


def data_gaps(df: pd.DataFrame, graph: nx.DiGraph):
    """Blind spots of the extract and the next data request that would close each one."""
    def example(mask):
        rows = df[mask].sort_values(["priority_score", "gid"], ascending=[False, True])
        return ";".join(str(x) for x in rows.gid.head(5))

    main_component = max(nx.weakly_connected_components(graph), key=len)
    gaps = [
        ("boundary_likely_continues", df.boundary & (df.p_hidden_outgoing >= .5),
         "узлы 4-го колена, у которых по аналогии с коленами 1–3 вероятны скрытые исходящие",
         "продлить обход на 1–2 колена от этих gid"),
        ("boundary_other", df.boundary & (df.p_hidden_outgoing < .5),
         "остальные узлы 4-го колена без исходящих",
         "исходящие переводы этих gid за июль 2026 выборочно, начиная с высокого приоритета"),
        ("seed_without_outgoing", df.is_seed & (df.out_deg == 0),
         "seed без исходящих в выгрузке: их роль по данным не установить",
         "входящие переводы и переводы в другие банки по этим seed"),
        ("gives_more_than_received", (df.out_kzt > df.in_kzt) & (df.in_kzt > 0),
         "отдали больше, чем видимо получили: источник денег вне выборки",
         "входящие переводы извне выборки для этих gid"),
        ("near_threshold", (df.near_threshold_in >= 3) & (df.near_threshold_in >= .5 * df.in_tx),
         "большинство входов у порога 5–10 тыс. KZT: возможно дробление ниже порога",
         "все переводы этих gid без порога 5 000 KZT"),
        ("outside_main_component", ~df.gid.isin(main_component),
         "узлы вне крупнейшей компоненты: связь с основной сетью не видна",
         "входящие переводы seed этих фрагментов, чтобы найти общие источники"),
    ]
    return pd.DataFrame([{"gap": key, "n_nodes": int(mask.sum()), "meaning": meaning,
                          "next_request": request, "example_gids": example(mask)}
                         for key, mask, meaning, request in gaps])


def write_outputs(df: pd.DataFrame, edges: pd.DataFrame, cycles: pd.DataFrame,
                  resilience: pd.DataFrame, gaps: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    node_cols = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence",
                 "depth", "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt",
                 "in_tx", "out_tx", "seed_reach", "bridge", "rapid_48h", "boundary",
                 "p_hidden_outgoing", "cycles", "cycle_with_seed", "sync_payers_max",
                 "repeat_amount_max", "near_threshold_in", "attention"]
    df[node_cols].to_csv(out / "nodes_roles.csv", index=False)
    cluster_of = dict(zip(df.gid, df.cluster_id))
    internal = defaultdict(float)
    for src, dst, amount in edges[["src", "dst", "sum_kzt"]].itertuples(index=False, name=None):
        if cluster_of[src] == cluster_of[dst]:
            internal[cluster_of[src]] += float(amount)
    clusters = []
    for cid, group in df.groupby("cluster_id", sort=True):
        leaders = group.sort_values(["priority_score", "gid"], ascending=[False, True]).head(5)
        counts = group.role.value_counts().to_dict()
        n_seed = int(group.is_seed.sum())
        if n_seed >= 2 and (counts.get("coordinator", 0) or counts.get("consolidator", 0)):
            purpose = "возможный общий контур сбора потоков от нескольких seed"
        elif counts.get("distributor", 0):
            purpose = "ветвь с веерным распределением исходящих переводов"
        elif counts.get("consolidator", 0):
            purpose = "локальный сбор средств от нескольких плательщиков"
        elif counts.get("transit", 0):
            purpose = "цепочка транзитных переводов"
        elif len(group) == 1 and n_seed:
            purpose = "seed без наблюдаемых связей"
        elif counts.get("terminal", 0):
            purpose = "ветвь к наблюдаемым конечным получателям"
        else:
            purpose = "периферийная или обрезанная границей выборки ветвь"
        hypothesis = (f"Гипотеза: {purpose}; {n_seed} seed, "
                      f"{counts.get('coordinator', 0)} координаторов, "
                      f"{counts.get('consolidator', 0)} сборщиков, "
                      f"{counts.get('distributor', 0)} распределителей. Требует проверки.")
        clusters.append({"cluster_id": int(cid), "n_nodes": len(group),
                         "n_seed": n_seed,
                         "sum_kzt_internal": round(internal[cid], 2),
                         "top_gids": ";".join(str(x) for x in leaders.gid),
                         "hypothesis": hypothesis})
    pd.DataFrame(clusters).to_csv(out / "clusters.csv", index=False)
    leaders = df.sort_values(["priority_score", "gid"], ascending=[False, True]).head(max(20, min(100, len(df)))).copy()
    leaders.insert(0, "rank", range(1, len(leaders) + 1))
    leaders["why"] = leaders.apply(
        lambda r: (f"{r.evidence} Приоритет: {r.in_deg} плательщиков, "
                   f"{r.in_kzt:,.0f} KZT входа, {r.out_deg} получателей, "
                   f"достижим из {r.seed_reach} seed, мост {r.bridge:.4f}; "
                   f"итог {r.priority_score:.3f}."), axis=1)
    leaders[["rank", "gid", "role", "priority_score", "why", "attention"]].to_csv(out / "top_nodes.csv", index=False)
    cycles.to_csv(out / "cycles.csv", index=False)
    resilience.to_csv(out / "resilience.csv", index=False)
    gaps.to_csv(out / "data_gaps.csv", index=False)

    # Readable in a browser without a local server or CDN.
    payload = {"nodes": [{"gid": str(r.gid), "role": r.role, "cluster": int(r.cluster_id),
                          "score": r.priority_score, "evidence": r.evidence,
                          "depth": int(r.depth), "seed": bool(r.is_seed),
                          "incoming": int(r.in_deg), "outgoing": int(r.out_deg),
                          "inKzt": float(r.in_kzt), "outKzt": float(r.out_kzt),
                          "seedReach": int(r.seed_reach), "boundary": bool(r.boundary),
                          "cycles": int(r.cycles), "syncPayers": int(r.sync_payers_max),
                          "pHidden": float(r.p_hidden_outgoing), "attention": r.attention}
                         for r in df.itertuples(index=False)],
               "edges": [{"src": str(r.src), "dst": str(r.dst), "amount": float(r.sum_kzt), "count": int(r.n_tx)}
                         for r in edges.itertuples(index=False)]}
    page = (ROOT / "viewer.html").read_text(encoding="utf-8")
    page = page.replace("/* GRAPH_DATA */ null", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    (out / "network.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "out")
    args = parser.parse_args()
    nodes, edges, tx = load_data(args.data)
    graph, df = graph_features(nodes, edges, tx)
    df, cycles = find_cycles(graph, df)
    df = flag_attention(rank_nodes(score_roles(assign_clusters(graph, df, edges))))
    resilience = network_resilience(graph, df)
    gaps = data_gaps(df, graph)
    write_outputs(df, edges, cycles, resilience, gaps, args.out)
    assert len(df) == len(nodes) and set(df.role) <= ROLES
    assert df.evidence.str.len().between(1, 200).all()
    print(f"Готово: {len(df)} узлов, {len(edges)} рёбер, {df.cluster_id.nunique()} кластеров")
    print(f"Роли: {df.role.value_counts().to_dict()}")
    print(f"Циклов до {CYCLE_MAX_LEN} переводов: {len(cycles)}; узлов с флагами: {(df.attention != 'нет').sum()}")
    print(f"Файлы: {args.out.resolve()}")


if __name__ == "__main__":
    main()
