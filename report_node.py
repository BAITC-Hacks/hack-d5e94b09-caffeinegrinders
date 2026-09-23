#!/usr/bin/env python3
"""Print a deterministic analyst report for one gid without using an LLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from graph_tools import GraphIndex, UnknownGid


def node_report(index: GraphIndex, gid: str) -> str:
    profile = index.node_profile(gid)
    parties = index.counterparties(gid, limit=8)
    routes = index.node_routes(gid, limit=5)
    cycles = index.node_cycles(gid, limit=5)
    lines = [
        f"gid {profile['gid']}",
        f"role: {profile['role']} ({profile['role_score']:.3f})",
        f"priority: #{profile['priority_rank']} score={profile['priority_score']:.3f}",
        f"cluster: {profile['cluster_id']} depth={profile['depth']} seed={profile['is_seed']}",
        f"flows: in {profile['in_kzt']:,.0f} KZT from {profile['in_deg']} payers; "
        f"out {profile['out_kzt']:,.0f} KZT to {profile['out_deg']} recipients",
        f"evidence: {profile['evidence']}",
        f"attention: {profile['attention']}",
        "",
        "direct counterparties:",
    ]
    for row in parties["shown"]:
        arrow = "from" if row["direction"] == "in" else "to"
        lines.append(f"  {arrow} {row['gid']}: {row['sum_kzt']:,.0f} KZT ({row['role']})")
    lines.append("")
    lines.append(f"routes through node: {routes['total']} ({routes['repeated']} repeated, {routes['chains']} chains)")
    for row in routes["shown"]:
        lines.append(f"  {row['kind']} {row['path']} · {row['forwarded_kzt']:,.0f} KZT")
    lines.append("")
    lines.append(f"cycles through node: {cycles['total']}")
    for row in cycles["shown"]:
        lines.append(f"  {row['path']} · bottleneck {row['bottleneck_kzt']:,.0f} KZT")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gid")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    try:
        print(node_report(GraphIndex(args.data), args.gid))
    except UnknownGid as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
