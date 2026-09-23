#!/usr/bin/env python3
"""Sanity-check the generated self-contained network.html payload."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PAYLOAD_RE = re.compile(r"const data = (\{.*?\});\nconst roleColors=", re.S)


def extract_payload(html: str) -> dict:
    if "/* GRAPH_DATA */ null" in html:
        raise ValueError("viewer still contains the graph-data placeholder")
    match = PAYLOAD_RE.search(html)
    if not match:
        raise ValueError("cannot find embedded viewer payload")
    return json.loads(match.group(1))


def check_viewer(path: Path) -> dict:
    payload = extract_payload(path.read_text(encoding="utf-8"))
    required = {"nodes", "edges", "cycles", "routes", "clusters", "clusterFlows", "clusterFlowSummary",
                "gaps", "riskFlags", "seedCoverage", "seedComponents", "components",
                "componentRoles", "componentAttention", "isolatedNodes", "amountBands",
                "roleSummary", "topEdges", "dailySummary", "boundaryReview",
                "clusterRoles", "seedRoleReach", "seedOverlap", "seedAttention",
                "attentionExamples", "clusterAttention",
                "roleAttention", "depthAttention", "attentionOverlap", "routeNodes",
                "cycleNodes", "depthSummary", "clusterDepths", "roleDepths",
                "roleFlows", "depthFlows", "topCounterparties"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"viewer payload missing keys: {sorted(missing)}")
    if not payload["nodes"] or not payload["edges"]:
        raise ValueError("viewer payload must contain nodes and edges")
    if not any(node.get("timeline") for node in payload["nodes"]):
        raise ValueError("viewer nodes must include timeline arrays")
    return {"nodes": len(payload["nodes"]), "edges": len(payload["edges"]),
            "routes": len(payload["routes"]), "clusters": len(payload["clusters"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path, nargs="?", default=Path("out/network.html"))
    args = parser.parse_args()
    result = check_viewer(args.html)
    print(f"OK: {result['nodes']} узлов, {result['edges']} рёбер, "
          f"{result['routes']} маршрутов, {result['clusters']} кластеров")


if __name__ == "__main__":
    main()
