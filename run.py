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
    df["bridge"] = df.gid.map(between).fillna(0.0)
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
    return graph, df


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
                reason = f"Граница обхода: колено 4, вход от {r.in_deg} клиентов; исходящие за пределом выгрузки неизвестны."
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


def write_outputs(df: pd.DataFrame, edges: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    node_cols = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence",
                 "depth", "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt",
                 "in_tx", "out_tx", "seed_reach", "bridge", "rapid_48h", "boundary"]
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
    leaders[["rank", "gid", "role", "priority_score", "why"]].to_csv(out / "top_nodes.csv", index=False)

    # Readable in a browser without a local server or CDN.
    payload = {"nodes": [{"gid": str(r.gid), "role": r.role, "cluster": int(r.cluster_id),
                          "score": r.priority_score, "evidence": r.evidence,
                          "depth": int(r.depth), "seed": bool(r.is_seed),
                          "incoming": int(r.in_deg), "outgoing": int(r.out_deg),
                          "inKzt": float(r.in_kzt), "outKzt": float(r.out_kzt),
                          "seedReach": int(r.seed_reach), "boundary": bool(r.boundary)}
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
    df = rank_nodes(score_roles(assign_clusters(graph, df, edges)))
    write_outputs(df, edges, args.out)
    assert len(df) == len(nodes) and set(df.role) <= ROLES
    assert df.evidence.str.len().between(1, 200).all()
    print(f"Готово: {len(df)} узлов, {len(edges)} рёбер, {df.cluster_id.nunique()} кластеров")
    print(f"Роли: {df.role.value_counts().to_dict()}")
    print(f"Файлы: {args.out.resolve()}")


if __name__ == "__main__":
    main()
