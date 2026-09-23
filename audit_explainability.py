#!/usr/bin/env python3
"""Sample nodes and print their role evidence and priority breakdown."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BREAKDOWN = ["priority_payers", "priority_incoming", "priority_seed_reach",
             "priority_bridge", "priority_recipients", "priority_rapid"]


def audit_text(out: Path, sample: int = 10) -> str:
    nodes = pd.read_csv(out / "nodes_roles.csv")
    top = nodes.sort_values(["priority_score", "gid"], ascending=[False, True]).head(sample)
    lines = ["Explainability audit", ""]
    for row in top.itertuples(index=False):
        parts = ", ".join(f"{name.replace('priority_', '')}={getattr(row, name):.3f}"
                          for name in BREAKDOWN if hasattr(row, name))
        lines += [
            f"{row.gid}",
            f"  role={row.role}, role_score={row.role_score:.3f}, priority={row.priority_score:.3f}",
            f"  evidence={row.evidence}",
            f"  breakdown: base={row.priority_base:.3f}; {parts}; "
            f"factors role={row.priority_role_factor:.2f}, boundary={row.priority_boundary_factor:.2f}, seed={row.priority_seed_factor:.2f}",
            f"  attention={row.attention}",
            "",
        ]
    return "\n".join(lines).rstrip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--sample", type=int, default=10)
    args = parser.parse_args()
    print(audit_text(args.out, args.sample))


if __name__ == "__main__":
    main()
