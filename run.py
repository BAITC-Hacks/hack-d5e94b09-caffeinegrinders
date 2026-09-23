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

ROLE_ORDER = ("coordinator", "consolidator", "distributor", "transit", "terminal", "peripheral")
ROLES = set(ROLE_ORDER)
ROOT = Path(__file__).resolve().parent
CYCLE_MAX_LEN = 4
EXTRACT_THRESHOLD_KZT = 5_000
NEAR_THRESHOLD_KZT = 10_000
SYNC_PAYERS_MIN = 3
REPEAT_AMOUNT_MIN = 4
REMOVAL_STEPS = (0, 5, 10, 20, 50, 100)
RELAY_MAX_LAG_DAYS = 2
RELAY_SHARE = (.5, 1.05)
STABLE_ROUTE_DAYS = 2
AMOUNT_BANDS = [
    (5_000, 10_000, "5k-10k"),
    (10_000, 50_000, "10k-50k"),
    (50_000, 100_000, "50k-100k"),
    (100_000, 500_000, "100k-500k"),
    (500_000, math.inf, "500k+"),
]


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
        if frame[list(columns)].isna().any().any():
            raise ValueError("Required fields must not be null")
    if nodes.empty:
        raise ValueError("nodes must not be empty")
    for frame, columns in ((nodes, ["gid", "depth"]),
                           (edges, ["src", "dst", "n_tx"]),
                           (tx, ["src", "dst"])):
        for column in columns:
            if not pd.api.types.is_integer_dtype(frame[column]):
                raise ValueError(f"{column} must use an integer dtype")
    if not pd.api.types.is_bool_dtype(nodes.is_seed):
        raise ValueError("is_seed must use a boolean dtype")
    if not nodes.depth.between(0, 4).all():
        raise ValueError("depth must be between 0 and 4")
    for frame in (edges, tx):
        if not pd.api.types.is_numeric_dtype(frame.sum_kzt) or not (
            np.isfinite(frame.sum_kzt) & (frame.sum_kzt > 0)
        ).all():
            raise ValueError("sum_kzt must be finite and positive")
    if not (edges.n_tx > 0).all():
        raise ValueError("n_tx must be positive")
    if nodes.gid.isna().any() or nodes.gid.duplicated().any():
        raise ValueError("nodes.gid must be unique and nonempty")
    if edges.duplicated(["src", "dst"]).any():
        raise ValueError("Duplicate aggregated edges")
    gids = set(nodes.gid)
    for frame in (edges, tx):
        if not set(frame.src).union(frame.dst).issubset(gids):
            raise ValueError("Edge or transaction endpoint absent from nodes")
    grouped = tx.groupby(["src", "dst"], as_index=False).agg(amount=("sum_kzt", "sum"), count=("sum_kzt", "size"))
    check = edges.merge(grouped, on=["src", "dst"], how="outer", indicator=True)
    if (check._merge != "both").any() or not np.allclose(check.sum_kzt, check.amount, rtol=0, atol=.01) or not (check.n_tx == check["count"]).all():
        raise ValueError("transactions and edges disagree")
    tx["date"] = pd.to_datetime(tx["date"], errors="raise")
    if tx.date.isna().any():
        raise ValueError("Transaction dates must not be empty")
    return (nodes.sort_values("gid").reset_index(drop=True),
            edges.sort_values(["src", "dst"]).reset_index(drop=True), tx)


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
    # The lower bound matters on extracts collected without the cut-off.
    near = tx[tx.sum_kzt.between(EXTRACT_THRESHOLD_KZT, NEAR_THRESHOLD_KZT, inclusive="left")].groupby("dst").size()
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


def daily_timeline(tx: pd.DataFrame):
    """Per-node daily incoming/outgoing activity for analyst drill-down."""
    columns = ["gid", "date", "in_tx", "out_tx", "in_kzt", "out_kzt",
               "net_kzt", "unique_payers", "unique_recipients"]
    if tx.empty:
        return pd.DataFrame(columns=columns)

    dated = tx.copy()
    dated["date"] = dated.date.dt.date.astype(str)
    incoming = dated.groupby(["dst", "date"], as_index=False).agg(
        in_tx=("sum_kzt", "size"),
        in_kzt=("sum_kzt", "sum"),
        unique_payers=("src", "nunique"),
    ).rename(columns={"dst": "gid"})
    outgoing = dated.groupby(["src", "date"], as_index=False).agg(
        out_tx=("sum_kzt", "size"),
        out_kzt=("sum_kzt", "sum"),
        unique_recipients=("dst", "nunique"),
    ).rename(columns={"src": "gid"})
    timeline = incoming.merge(outgoing, on=["gid", "date"], how="outer").fillna(0)
    for column in ("in_tx", "out_tx", "unique_payers", "unique_recipients"):
        timeline[column] = timeline[column].astype(int)
    for column in ("in_kzt", "out_kzt"):
        timeline[column] = timeline[column].round(2)
    timeline["net_kzt"] = (timeline.in_kzt - timeline.out_kzt).round(2)
    return timeline[columns].sort_values(["gid", "date"]).reset_index(drop=True)


def daily_summary(tx: pd.DataFrame):
    """Network-level daily visible transaction activity."""
    columns = ["date", "n_tx", "sum_kzt", "unique_senders", "unique_recipients", "active_nodes"]
    if tx.empty:
        return pd.DataFrame(columns=columns)
    dated = tx.copy()
    dated["date"] = dated.date.dt.date.astype(str)
    rows = []
    for date, group in dated.groupby("date", sort=True):
        active = pd.concat([group.src, group.dst], ignore_index=True).nunique()
        rows.append({"date": date,
                     "n_tx": int(len(group)),
                     "sum_kzt": round(float(group.sum_kzt.sum()), 2),
                     "unique_senders": int(group.src.nunique()),
                     "unique_recipients": int(group.dst.nunique()),
                     "active_nodes": int(active)})
    return pd.DataFrame(rows, columns=columns)


def find_routes(tx: pd.DataFrame, df: pd.DataFrame):
    """Forwarding routes based on timing and similar transfer amounts.

    A repeated route A -> B -> C means B forwarded 50-105% of an incoming
    transfer from A to C within 0-2 days on at least two different forwarding
    days. A chain A -> B -> C -> D links two such relay episodes through the
    same B -> C transfer. This is a structural clue, not proof that the exact
    same money moved.
    """
    route_columns = ["route_id", "kind", "hops", "path", "relay_days",
                     "forwarded_kzt", "first_date", "last_date", "n_seed"]
    if tx.empty:
        df["relay_routes"] = 0
        df["chain_transits"] = 0
        return df, pd.DataFrame(columns=route_columns)

    tx = tx.reset_index(drop=True).rename_axis("tx_id").reset_index()
    hop_in = tx.rename(columns={"tx_id": "in_id", "src": "a", "dst": "b",
                                "date": "d1", "sum_kzt": "s1"})
    hop_out = tx.rename(columns={"tx_id": "out_id", "src": "b", "dst": "c",
                                 "date": "d2", "sum_kzt": "s2"})
    relays = hop_in.merge(hop_out, on="b")
    lag = (relays.d2 - relays.d1).dt.days
    relays = relays[(relays.a != relays.c) & lag.between(0, RELAY_MAX_LAG_DAYS)
                    & relays.s2.between(RELAY_SHARE[0] * relays.s1, RELAY_SHARE[1] * relays.s1)]

    nxt = relays.rename(columns={"in_id": "out_id", "a": "b", "b": "c", "c": "d",
                                 "d1": "d2", "s1": "s2", "out_id": "last_id",
                                 "d2": "d3", "s2": "s3"})
    chains = relays.merge(nxt, on=["out_id", "b", "c", "d2", "s2"])
    chains = chains[chains.d != chains.a]

    def summarise(frame, nodes, last_id, last_sum, last_date, kind):
        rows = []
        for key, group in frame.groupby(nodes, sort=False):
            forwarded = group.drop_duplicates(last_id)[last_sum].sum()
            rows.append({"kind": kind, "hops": len(nodes) - 1,
                         "path": " → ".join(str(node) for node in key),
                         "relay_days": group[last_date].dt.date.nunique(),
                         "forwarded_kzt": round(float(forwarded), 2),
                         "first_date": group.d1.min().date().isoformat(),
                         "last_date": group[last_date].max().date().isoformat(),
                         "nodes": key})
        return rows

    rows = summarise(relays, ["a", "b", "c"], "out_id", "s2", "d2", "repeated")
    rows = [row for row in rows if row["relay_days"] >= STABLE_ROUTE_DAYS]
    rows += summarise(chains, ["a", "b", "c", "d"], "last_id", "s3", "d3", "chain")
    seeds = set(df.gid[df.is_seed])
    relay_middle, chain_inner = Counter(), Counter()
    for row in rows:
        inner = row["nodes"][1:-1]
        row["n_seed"] = sum(node in seeds for node in row["nodes"])
        (relay_middle if row["kind"] == "repeated" else chain_inner).update(inner)
    routes = pd.DataFrame(rows, columns=["kind", "hops", "path", "relay_days",
                                         "forwarded_kzt", "first_date", "last_date",
                                         "n_seed"])
    routes = routes.sort_values(["kind", "relay_days", "forwarded_kzt", "path"],
                                ascending=[False, False, False, True]).reset_index(drop=True)
    routes.insert(0, "route_id", range(1, len(routes) + 1))
    df["relay_routes"] = df.gid.map(relay_middle).fillna(0).astype(int)
    df["chain_transits"] = df.gid.map(chain_inner).fillna(0).astype(int)
    return df, routes[route_columns]


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
        if r.relay_routes:
            items.append(f"повторная пересылка за ≤2 дня ({r.relay_routes} маршр.)")
        elif r.chain_transits:
            items.append(f"звено сквозной цепочки ({r.chain_transits})")
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
    contributions = {
        "priority_payers": .20 * signals["payers"],
        "priority_incoming": .20 * signals["incoming"],
        "priority_seed_reach": .20 * signals["seeds"],
        "priority_bridge": .20 * signals["bridge"],
        "priority_recipients": .15 * signals["recipients"],
        "priority_rapid": .05 * df.rapid_48h,
    }
    for column, values in contributions.items():
        df[column] = values.round(6)
    base = sum(contributions.values())
    df["priority_base"] = base.round(6)
    df["priority_role_factor"] = df.role.map({"coordinator": 1.0, "consolidator": 1.0,
                                              "distributor": 1.0, "transit": .95,
                                              "terminal": .85, "peripheral": .75})
    df["priority_boundary_factor"] = np.where(df.boundary, .65, 1.0)
    df["priority_seed_factor"] = np.where(df.is_seed, .85, 1.0)
    score = base * df.priority_role_factor * df.priority_boundary_factor * df.priority_seed_factor
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
    # On graphs smaller than the largest step, report what was actually removed.
    steps = sorted({min(step, graph.number_of_nodes()) for step in REMOVAL_STEPS})
    for strategy, order in (("priority", by_priority), ("random", at_random)):
        for n in steps:
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


def risk_flags_summary(df: pd.DataFrame):
    """Counts of optional AML attention flags, separate from roles and priority."""
    def examples(mask):
        rows = df[mask].sort_values(["priority_score", "gid"], ascending=[False, True])
        return ";".join(str(x) for x in rows.gid.head(5))

    flags = [
        ("cycle", df.cycles > 0, "узел входит в направленный возвратный поток"),
        ("relay_route", df.relay_routes > 0, "узел является посредником устойчивого маршрута пересылки"),
        ("chain_transit", df.chain_transits > 0, "узел является внутренним звеном сквозной цепочки"),
        ("sync_payers", df.sync_payers_max >= SYNC_PAYERS_MIN,
         "несколько разных плательщиков в один день"),
        ("repeat_amount", df.repeat_amount_max >= REPEAT_AMOUNT_MIN,
         "одна и та же сумма отправлялась многократно"),
        ("near_threshold", (df.near_threshold_in >= 3) & (df.near_threshold_in >= .5 * df.in_tx),
         "много входящих переводов у порога 5-10 тыс. KZT"),
        ("boundary_likely_continues", df.boundary & (df.p_hidden_outgoing >= .5),
         "граничный узел с вероятным скрытым исходящим продолжением"),
        ("gives_more_than_received", (df.out_kzt > df.in_kzt) & (df.in_kzt > 0),
         "исходящий поток выше видимого входа"),
    ]
    total = max(len(df), 1)
    return pd.DataFrame([{"flag": key, "n_nodes": int(mask.sum()),
                          "share": round(float(mask.sum()) / total, 4),
                          "meaning": meaning, "example_gids": examples(mask)}
                         for key, mask, meaning in flags])


def attention_examples(df: pd.DataFrame):
    """One row per node and optional attention flag for transparent review."""
    columns = ["flag", "gid", "role", "cluster_id", "priority_score",
               "metric_value", "meaning", "evidence", "attention"]
    flags = [
        ("cycle", df.cycles > 0, df.cycles, "узел входит в направленный возвратный поток"),
        ("relay_route", df.relay_routes > 0, df.relay_routes,
         "узел является посредником устойчивого маршрута пересылки"),
        ("chain_transit", df.chain_transits > 0, df.chain_transits,
         "узел является внутренним звеном сквозной цепочки"),
        ("sync_payers", df.sync_payers_max >= SYNC_PAYERS_MIN, df.sync_payers_max,
         "несколько разных плательщиков в один день"),
        ("repeat_amount", df.repeat_amount_max >= REPEAT_AMOUNT_MIN, df.repeat_amount_max,
         "одна и та же сумма отправлялась многократно"),
        ("near_threshold", (df.near_threshold_in >= 3) & (df.near_threshold_in >= .5 * df.in_tx),
         df.near_threshold_in, "много входящих переводов у порога 5-10 тыс. KZT"),
        ("boundary_likely_continues", df.boundary & (df.p_hidden_outgoing >= .5),
         df.p_hidden_outgoing, "граничный узел с вероятным скрытым исходящим продолжением"),
        ("gives_more_than_received", (df.out_kzt > df.in_kzt) & (df.in_kzt > 0),
         df.out_kzt - df.in_kzt, "исходящий поток выше видимого входа"),
    ]
    rows = []
    for flag, mask, values, meaning in flags:
        group = df[mask].copy()
        if group.empty:
            continue
        group["metric_value"] = values[mask].round(6)
        for row in group.sort_values(["priority_score", "gid"], ascending=[False, True]).itertuples(index=False):
            rows.append({"flag": flag,
                         "gid": int(row.gid),
                         "role": row.role,
                         "cluster_id": int(row.cluster_id),
                         "priority_score": round(float(row.priority_score), 6),
                         "metric_value": float(row.metric_value),
                         "meaning": meaning,
                         "evidence": row.evidence,
                         "attention": row.attention})
    return pd.DataFrame(rows, columns=columns)


def cluster_flows(df: pd.DataFrame, edges: pd.DataFrame):
    """Aggregated transfers between clusters for ingress/egress review."""
    cluster_of = dict(zip(df.gid, df.cluster_id))
    rows = []
    for row in edges.itertuples(index=False):
        src_cluster = int(cluster_of[row.src])
        dst_cluster = int(cluster_of[row.dst])
        if src_cluster != dst_cluster:
            rows.append({"src_cluster": src_cluster, "dst_cluster": dst_cluster,
                         "sum_kzt": float(row.sum_kzt), "n_edges": 1,
                         "n_tx": int(row.n_tx)})
    if not rows:
        return pd.DataFrame(columns=["src_cluster", "dst_cluster", "sum_kzt", "n_edges", "n_tx"])
    flows = pd.DataFrame(rows).groupby(["src_cluster", "dst_cluster"], as_index=False).agg(
        sum_kzt=("sum_kzt", "sum"),
        n_edges=("n_edges", "sum"),
        n_tx=("n_tx", "sum"),
    )
    flows["sum_kzt"] = flows.sum_kzt.round(2)
    return flows.sort_values(["sum_kzt", "src_cluster", "dst_cluster"],
                             ascending=[False, True, True]).reset_index(drop=True)


def seed_coverage(graph: nx.DiGraph, df: pd.DataFrame):
    """Per-seed reachability and direct outgoing footprint."""
    rows = []
    lookup = df.set_index("gid")
    for seed in df.loc[df.is_seed, "gid"].sort_values():
        seen = nx.single_source_shortest_path_length(graph, int(seed), cutoff=4)
        out_edges = list(graph.out_edges(int(seed), data=True))
        rows.append({"seed_gid": int(seed),
                     "cluster_id": int(lookup.loc[seed, "cluster_id"]),
                     "reachable_nodes": len(seen) - 1,
                     "direct_recipients": len(out_edges),
                     "direct_out_kzt": round(sum(data["sum_kzt"] for _, _, data in out_edges), 2),
                     "max_depth_reached": max(seen.values(), default=0)})
    return pd.DataFrame(rows, columns=["seed_gid", "cluster_id", "reachable_nodes",
                                       "direct_recipients", "direct_out_kzt",
                                       "max_depth_reached"])


def component_summary(graph: nx.DiGraph, df: pd.DataFrame, edges: pd.DataFrame):
    """Weakly connected network fragments with seeds, turnover and leaders."""
    rows = []
    components = sorted(nx.weakly_connected_components(graph), key=lambda c: (-len(c), min(c)))
    for idx, members in enumerate(components, start=1):
        group = df[df.gid.isin(members)]
        internal = edges[edges.src.isin(members) & edges.dst.isin(members)]
        leaders = group.sort_values(["priority_score", "gid"], ascending=[False, True]).head(5)
        rows.append({"component_id": idx,
                     "n_nodes": int(len(group)),
                     "n_seed": int(group.is_seed.sum()),
                     "sum_kzt_internal": round(float(internal.sum_kzt.sum()), 2),
                     "n_edges_internal": int(len(internal)),
                     "top_gids": ";".join(str(x) for x in leaders.gid),
                     "top_priority": round(float(leaders.priority_score.max()), 6) if len(leaders) else 0.0,
                     "is_main_component": idx == 1})
    return pd.DataFrame(rows)


def amount_band_summary(tx: pd.DataFrame):
    """Transaction amount distribution in fixed, explainable KZT bands."""
    columns = ["amount_band", "n_tx", "sum_kzt", "share_tx", "share_kzt"]
    if tx.empty:
        return pd.DataFrame(columns=columns)
    total_tx = len(tx)
    total_kzt = float(tx.sum_kzt.sum())
    rows = []
    for low, high, label in AMOUNT_BANDS:
        mask = (tx.sum_kzt >= low) & (tx.sum_kzt < high)
        amount = float(tx.loc[mask, "sum_kzt"].sum())
        rows.append({"amount_band": label, "n_tx": int(mask.sum()),
                     "sum_kzt": round(amount, 2),
                     "share_tx": round(float(mask.sum()) / total_tx, 4),
                     "share_kzt": round(amount / total_kzt, 4) if total_kzt else 0.0})
    return pd.DataFrame(rows, columns=columns)


def role_summary(df: pd.DataFrame):
    """Aggregate explainable role output without changing role assignments."""
    columns = ["role", "n_nodes", "share_nodes", "n_seed", "boundary_nodes",
               "avg_role_score", "avg_priority_score", "in_kzt", "out_kzt",
               "n_attention", "top_gids"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    total = len(df)
    for role in ROLE_ORDER:
        group = df[df.role == role]
        leaders = group.sort_values(["priority_score", "gid"], ascending=[False, True]).head(5)
        rows.append({"role": role,
                     "n_nodes": int(len(group)),
                     "share_nodes": round(len(group) / total, 4),
                     "n_seed": int(group.is_seed.sum()) if len(group) else 0,
                     "boundary_nodes": int(group.boundary.sum()) if len(group) else 0,
                     "avg_role_score": round(float(group.role_score.mean()), 6) if len(group) else 0.0,
                     "avg_priority_score": round(float(group.priority_score.mean()), 6) if len(group) else 0.0,
                     "in_kzt": round(float(group.in_kzt.sum()), 2) if len(group) else 0.0,
                     "out_kzt": round(float(group.out_kzt.sum()), 2) if len(group) else 0.0,
                     "n_attention": int((group.attention != "нет").sum()) if len(group) else 0,
                     "top_gids": ";".join(str(x) for x in leaders.gid)})
    return pd.DataFrame(rows, columns=columns)


def top_edge_summary(edges: pd.DataFrame, df: pd.DataFrame, limit: int = 200):
    """Largest visible transfer pairs with endpoint roles for manual review."""
    columns = ["rank", "src", "dst", "sum_kzt", "n_tx", "src_role", "dst_role",
               "src_cluster", "dst_cluster", "src_seed", "dst_seed", "same_cluster"]
    if edges.empty:
        return pd.DataFrame(columns=columns)
    meta = df.set_index("gid")
    rows = []
    top_edges = edges.sort_values(["sum_kzt", "src", "dst"], ascending=[False, True, True]).head(limit)
    for rank, edge in enumerate(top_edges.itertuples(index=False), start=1):
        src = meta.loc[edge.src]
        dst = meta.loc[edge.dst]
        src_cluster = int(src.cluster_id)
        dst_cluster = int(dst.cluster_id)
        rows.append({"rank": rank,
                     "src": int(edge.src),
                     "dst": int(edge.dst),
                     "sum_kzt": round(float(edge.sum_kzt), 2),
                     "n_tx": int(edge.n_tx),
                     "src_role": src.role,
                     "dst_role": dst.role,
                     "src_cluster": src_cluster,
                     "dst_cluster": dst_cluster,
                     "src_seed": bool(src.is_seed),
                     "dst_seed": bool(dst.is_seed),
                     "same_cluster": src_cluster == dst_cluster})
    return pd.DataFrame(rows, columns=columns)


def boundary_review(df: pd.DataFrame):
    """Queue boundary nodes for optional next-hop data requests."""
    columns = ["rank", "gid", "depth", "in_deg", "in_kzt", "p_hidden_outgoing",
               "cluster_id", "priority_score", "attention", "next_request"]
    boundary = df[df.boundary].copy()
    if boundary.empty:
        return pd.DataFrame(columns=columns)
    boundary = boundary.sort_values(["p_hidden_outgoing", "in_kzt", "in_deg", "gid"],
                                    ascending=[False, False, False, True]).reset_index(drop=True)
    boundary.insert(0, "rank", range(1, len(boundary) + 1))
    boundary["next_request"] = boundary.apply(
        lambda r: (f"Запросить исходящие переводы от gid {int(r.gid)} за следующий шаг; "
                   f"оценка скрытого выхода {r.p_hidden_outgoing:.0%}."), axis=1)
    return boundary[columns]


def cluster_role_matrix(df: pd.DataFrame):
    """Role composition inside each detected cluster."""
    columns = ["cluster_id", "role", "n_nodes", "share_cluster", "in_kzt",
               "out_kzt", "avg_priority_score", "top_gids"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    cluster_sizes = df.groupby("cluster_id").size().to_dict()
    for (cluster_id, role), group in df.groupby(["cluster_id", "role"], sort=True):
        leaders = group.sort_values(["priority_score", "gid"], ascending=[False, True]).head(5)
        rows.append({"cluster_id": int(cluster_id),
                     "role": role,
                     "n_nodes": int(len(group)),
                     "share_cluster": round(len(group) / cluster_sizes[cluster_id], 4),
                     "in_kzt": round(float(group.in_kzt.sum()), 2),
                     "out_kzt": round(float(group.out_kzt.sum()), 2),
                     "avg_priority_score": round(float(group.priority_score.mean()), 6),
                     "top_gids": ";".join(str(x) for x in leaders.gid)})
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["cluster_id", "n_nodes", "role"], ascending=[True, False, True]
    ).reset_index(drop=True)


def seed_role_reach(graph: nx.DiGraph, df: pd.DataFrame):
    """Per-seed reachable role mix within the documented four-transfer horizon."""
    role_cols = [f"n_{role}" for role in ROLE_ORDER]
    columns = ["seed_gid", "cluster_id", "reachable_nodes", "max_depth_reached",
               *role_cols, "boundary_nodes", "attention_nodes", "top_reachable_gids"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    lookup = df.set_index("gid")
    rows = []
    for seed in df.loc[df.is_seed, "gid"].sort_values():
        seen = nx.single_source_shortest_path_length(graph, int(seed), cutoff=4)
        reachable = sorted(node for node in seen if node != int(seed))
        group = lookup.loc[reachable] if reachable else df.iloc[:0]
        counts = group.role.value_counts().to_dict() if len(group) else {}
        leaders = group.sort_values(["priority_score", "gid"], ascending=[False, True]).head(5)
        row = {"seed_gid": int(seed),
               "cluster_id": int(lookup.loc[seed, "cluster_id"]),
               "reachable_nodes": int(len(reachable)),
               "max_depth_reached": max(seen.values(), default=0),
               "boundary_nodes": int(group.boundary.sum()) if len(group) else 0,
               "attention_nodes": int((group.attention != "нет").sum()) if len(group) else 0,
               "top_reachable_gids": ";".join(str(int(x)) for x in leaders.index)}
        row.update({f"n_{role}": int(counts.get(role, 0)) for role in ROLE_ORDER})
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def route_node_membership(routes: pd.DataFrame, df: pd.DataFrame):
    """Explode route paths into one row per route node and position."""
    columns = ["route_id", "kind", "position", "gid", "role", "cluster_id",
               "is_seed", "priority_score", "path"]
    if routes.empty:
        return pd.DataFrame(columns=columns)
    lookup = df.set_index("gid")
    rows = []
    for route in routes.itertuples(index=False):
        path = [int(part.strip()) for part in route.path.split("→")]
        for position, gid in enumerate(path, start=1):
            meta = lookup.loc[gid]
            rows.append({"route_id": int(route.route_id),
                         "kind": route.kind,
                         "position": position,
                         "gid": gid,
                         "role": meta.role,
                         "cluster_id": int(meta.cluster_id),
                         "is_seed": bool(meta.is_seed),
                         "priority_score": round(float(meta.priority_score), 6),
                         "path": route.path})
    return pd.DataFrame(rows, columns=columns)


def cycle_node_membership(cycles: pd.DataFrame, df: pd.DataFrame):
    """Explode cycle paths into one row per cycle node and position."""
    columns = ["cycle_id", "position", "gid", "role", "cluster_id", "is_seed",
               "priority_score", "bottleneck_kzt", "path"]
    if cycles.empty:
        return pd.DataFrame(columns=columns)
    lookup = df.set_index("gid")
    rows = []
    for cycle in cycles.itertuples(index=False):
        path = [int(part.strip()) for part in cycle.path.split("→")]
        if len(path) > 1 and path[0] == path[-1]:
            path = path[:-1]
        for position, gid in enumerate(path, start=1):
            meta = lookup.loc[gid]
            rows.append({"cycle_id": int(cycle.cycle_id),
                         "position": position,
                         "gid": gid,
                         "role": meta.role,
                         "cluster_id": int(meta.cluster_id),
                         "is_seed": bool(meta.is_seed),
                         "priority_score": round(float(meta.priority_score), 6),
                         "bottleneck_kzt": round(float(cycle.bottleneck_kzt), 2),
                         "path": cycle.path})
    return pd.DataFrame(rows, columns=columns)


def depth_summary(df: pd.DataFrame):
    """Overview by traversal depth, including the documented boundary layer."""
    columns = ["depth", "n_nodes", "share_nodes", "n_seed", "boundary_nodes",
               "in_kzt", "out_kzt", "avg_priority_score", "attention_nodes",
               "top_gids"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    total = len(df)
    for depth, group in df.groupby("depth", sort=True):
        leaders = group.sort_values(["priority_score", "gid"], ascending=[False, True]).head(5)
        rows.append({"depth": int(depth),
                     "n_nodes": int(len(group)),
                     "share_nodes": round(len(group) / total, 4),
                     "n_seed": int(group.is_seed.sum()),
                     "boundary_nodes": int(group.boundary.sum()),
                     "in_kzt": round(float(group.in_kzt.sum()), 2),
                     "out_kzt": round(float(group.out_kzt.sum()), 2),
                     "avg_priority_score": round(float(group.priority_score.mean()), 6),
                     "attention_nodes": int((group.attention != "нет").sum()),
                     "top_gids": ";".join(str(x) for x in leaders.gid)})
    return pd.DataFrame(rows, columns=columns)


def write_outputs(df: pd.DataFrame, edges: pd.DataFrame, cycles: pd.DataFrame,
                  routes: pd.DataFrame, resilience: pd.DataFrame, gaps: pd.DataFrame,
                  risk_flags: pd.DataFrame, flows: pd.DataFrame,
                  seed_report: pd.DataFrame, components: pd.DataFrame,
                  amount_bands: pd.DataFrame, roles: pd.DataFrame,
                  top_edges: pd.DataFrame, daily: pd.DataFrame,
                  boundary_queue: pd.DataFrame, cluster_roles: pd.DataFrame,
                  seed_roles: pd.DataFrame, attention_rows: pd.DataFrame,
                  route_nodes: pd.DataFrame, cycle_nodes: pd.DataFrame,
                  depths: pd.DataFrame, timeline: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    node_cols = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence",
                 "depth", "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt",
                 "in_tx", "out_tx", "seed_reach", "bridge", "rapid_48h", "boundary",
                 "p_hidden_outgoing", "cycles", "cycle_with_seed", "sync_payers_max",
                 "repeat_amount_max", "near_threshold_in", "relay_routes", "chain_transits",
                 "priority_base",
                 "priority_payers", "priority_incoming", "priority_seed_reach",
                 "priority_bridge", "priority_recipients", "priority_rapid",
                 "priority_role_factor", "priority_boundary_factor", "priority_seed_factor",
                 "attention"]
    df[node_cols].to_csv(out / "nodes_roles.csv", index=False)
    cluster_of = dict(zip(df.gid, df.cluster_id))
    internal = defaultdict(float)
    for src, dst, amount in edges[["src", "dst", "sum_kzt"]].itertuples(index=False, name=None):
        if cluster_of[src] == cluster_of[dst]:
            internal[cluster_of[src]] += float(amount)
    clusters = []
    cluster_payload = []
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
        cluster_payload.append({"id": int(cid), "nNodes": int(len(group)), "nSeed": n_seed,
                                "sumKztInternal": round(internal[cid], 2),
                                "topGids": [str(x) for x in leaders.gid],
                                "roles": {k: int(v) for k, v in counts.items()},
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
    routes.to_csv(out / "routes.csv", index=False)
    resilience.to_csv(out / "resilience.csv", index=False)
    gaps.to_csv(out / "data_gaps.csv", index=False)
    risk_flags.to_csv(out / "risk_flags.csv", index=False)
    flows.to_csv(out / "cluster_flows.csv", index=False)
    seed_report.to_csv(out / "seed_coverage.csv", index=False)
    components.to_csv(out / "components.csv", index=False)
    amount_bands.to_csv(out / "amount_bands.csv", index=False)
    roles.to_csv(out / "role_summary.csv", index=False)
    top_edges.to_csv(out / "top_edges.csv", index=False)
    daily.to_csv(out / "daily_summary.csv", index=False)
    boundary_queue.to_csv(out / "boundary_review.csv", index=False)
    cluster_roles.to_csv(out / "cluster_roles.csv", index=False)
    seed_roles.to_csv(out / "seed_role_reach.csv", index=False)
    attention_rows.to_csv(out / "attention_examples.csv", index=False)
    route_nodes.to_csv(out / "route_nodes.csv", index=False)
    cycle_nodes.to_csv(out / "cycle_nodes.csv", index=False)
    depths.to_csv(out / "depth_summary.csv", index=False)
    timeline.to_csv(out / "timeline.csv", index=False)

    timeline_by_gid = defaultdict(list)
    for r in timeline.itertuples(index=False):
        timeline_by_gid[int(r.gid)].append({
            "date": r.date, "inTx": int(r.in_tx), "outTx": int(r.out_tx),
            "inKzt": float(r.in_kzt), "outKzt": float(r.out_kzt),
            "netKzt": float(r.net_kzt),
            "payers": int(r.unique_payers), "recipients": int(r.unique_recipients),
        })

    # Readable in a browser without a local server or CDN.
    payload = {"nodes": [{"gid": str(r.gid), "role": r.role, "cluster": int(r.cluster_id),
                          "score": r.priority_score, "evidence": r.evidence,
                          "depth": int(r.depth), "seed": bool(r.is_seed),
                          "incoming": int(r.in_deg), "outgoing": int(r.out_deg),
                          "inKzt": float(r.in_kzt), "outKzt": float(r.out_kzt),
                          "seedReach": int(r.seed_reach), "boundary": bool(r.boundary),
                          "cycles": int(r.cycles), "syncPayers": int(r.sync_payers_max),
                          "nearThresholdIn": int(r.near_threshold_in),
                          "relayRoutes": int(r.relay_routes),
                          "chainTransits": int(r.chain_transits),
                          "inTx": int(r.in_tx), "outTx": int(r.out_tx),
                          "pHidden": float(r.p_hidden_outgoing), "attention": r.attention,
                          "priority": {
                              "base": float(r.priority_base),
                              "payers": float(r.priority_payers),
                              "incoming": float(r.priority_incoming),
                              "seedReach": float(r.priority_seed_reach),
                              "bridge": float(r.priority_bridge),
                              "recipients": float(r.priority_recipients),
                              "rapid": float(r.priority_rapid),
                              "roleFactor": float(r.priority_role_factor),
                              "boundaryFactor": float(r.priority_boundary_factor),
                              "seedFactor": float(r.priority_seed_factor),
                          },
                          "timeline": timeline_by_gid[int(r.gid)]}
                         for r in df.itertuples(index=False)],
               "edges": [{"src": str(r.src), "dst": str(r.dst), "amount": float(r.sum_kzt), "count": int(r.n_tx)}
                         for r in edges.itertuples(index=False)],
               "cycles": cycles.to_dict(orient="records"),
               "routes": routes.to_dict(orient="records"),
               "clusterFlows": flows.to_dict(orient="records"),
               "clusters": cluster_payload,
               "gaps": gaps.to_dict(orient="records"),
               "riskFlags": risk_flags.to_dict(orient="records"),
               "seedCoverage": seed_report.to_dict(orient="records"),
               "components": components.to_dict(orient="records"),
               "amountBands": amount_bands.to_dict(orient="records"),
               "roleSummary": roles.to_dict(orient="records"),
               "topEdges": top_edges.to_dict(orient="records"),
               "dailySummary": daily.to_dict(orient="records"),
               "boundaryReview": boundary_queue.to_dict(orient="records"),
               "clusterRoles": cluster_roles.to_dict(orient="records"),
               "seedRoleReach": seed_roles.to_dict(orient="records"),
               "attentionExamples": attention_rows.to_dict(orient="records"),
               "routeNodes": route_nodes.to_dict(orient="records"),
               "cycleNodes": cycle_nodes.to_dict(orient="records"),
               "depthSummary": depths.to_dict(orient="records")}
    page = (ROOT / "viewer.html").read_text(encoding="utf-8")
    page = page.replace("/* GRAPH_DATA */ null", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    (out / "network.html").write_text(page, encoding="utf-8")


def build_model(data: Path, include_transactions: bool = False):
    """Graph, per-node roles and metrics, and cycles; shared by the CLI and the assistant."""
    nodes, edges, tx = load_data(data)
    graph, df = graph_features(nodes, edges, tx)
    df, cycles = find_cycles(graph, df)
    df, routes = find_routes(tx, df)
    df = flag_attention(rank_nodes(score_roles(assign_clusters(graph, df, edges))))
    timeline = daily_timeline(tx)
    if include_transactions:
        return nodes, edges, graph, df, cycles, routes, timeline, tx
    return nodes, edges, graph, df, cycles, routes, timeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "out")
    args = parser.parse_args()
    nodes, edges, graph, df, cycles, routes, timeline, tx = build_model(args.data, include_transactions=True)
    resilience = network_resilience(graph, df)
    gaps = data_gaps(df, graph)
    risk_flags = risk_flags_summary(df)
    flows = cluster_flows(df, edges)
    seed_report = seed_coverage(graph, df)
    components = component_summary(graph, df, edges)
    amount_bands = amount_band_summary(tx)
    roles = role_summary(df)
    top_edges = top_edge_summary(edges, df)
    daily = daily_summary(tx)
    boundary_queue = boundary_review(df)
    cluster_roles = cluster_role_matrix(df)
    seed_roles = seed_role_reach(graph, df)
    attention_rows = attention_examples(df)
    route_nodes = route_node_membership(routes, df)
    cycle_nodes = cycle_node_membership(cycles, df)
    depths = depth_summary(df)
    write_outputs(df, edges, cycles, routes, resilience, gaps, risk_flags, flows,
                  seed_report, components, amount_bands, roles, top_edges,
                  daily, boundary_queue, cluster_roles, seed_roles, attention_rows,
                  route_nodes, cycle_nodes, depths, timeline, args.out)
    assert len(df) == len(nodes) and set(df.role) <= ROLES
    assert df.evidence.str.len().between(1, 200).all()
    print(f"Готово: {len(df)} узлов, {len(edges)} рёбер, {df.cluster_id.nunique()} кластеров")
    print(f"Роли: {df.role.value_counts().to_dict()}")
    print(f"Циклов до {CYCLE_MAX_LEN} переводов: {len(cycles)}; узлов с флагами: {(df.attention != 'нет').sum()}")
    print(f"Маршрутов пересылки: {(routes.kind == 'repeated').sum()} повторных, {(routes.kind == 'chain').sum()} цепочек")
    print(f"Дней активности в timeline.csv: {len(timeline)}")
    print(f"Файлы: {args.out.resolve()}")


if __name__ == "__main__":
    main()
