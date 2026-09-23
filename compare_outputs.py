#!/usr/bin/env python3
"""Compare two pipeline output folders and rank the most important changes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_nodes(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "nodes_roles.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    required = {"gid", "role", "cluster_id", "priority_score", "evidence", "attention"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["gid"] = df.gid.astype(str)
    return df


def compare_outputs(old_dir: Path, new_dir: Path) -> pd.DataFrame:
    old = load_nodes(old_dir).add_suffix("_old").rename(columns={"gid_old": "gid"})
    new = load_nodes(new_dir).add_suffix("_new").rename(columns={"gid_new": "gid"})
    merged = old.merge(new, on="gid", how="outer", indicator=True)
    merged["status"] = merged._merge.map({"left_only": "removed", "right_only": "new", "both": "kept"})
    merged["priority_delta"] = (
        merged.priority_score_new.fillna(0) - merged.priority_score_old.fillna(0)
    ).round(6)
    merged["role_changed"] = (
        merged.role_old.notna() & merged.role_new.notna() & (merged.role_old != merged.role_new)
    )
    merged["cluster_changed"] = (
        merged.cluster_id_old.notna() & merged.cluster_id_new.notna()
        & (merged.cluster_id_old != merged.cluster_id_new)
    )
    merged["attention_changed"] = (
        merged.attention_old.fillna("") != merged.attention_new.fillna("")
    )
    changed = merged[
        (merged.status != "kept")
        | merged.role_changed
        | merged.cluster_changed
        | merged.attention_changed
        | (merged.priority_delta.abs() >= .05)
    ].copy()
    changed["impact"] = changed.priority_delta.abs()
    changed["impact"] += changed.role_changed.astype(float) * .5
    changed["impact"] += changed.status.ne("kept").astype(float)
    changed = changed.sort_values(["impact", "gid"], ascending=[False, True])
    columns = ["gid", "status", "role_old", "role_new", "priority_score_old",
               "priority_score_new", "priority_delta", "cluster_id_old",
               "cluster_id_new", "role_changed", "cluster_changed",
               "attention_changed", "evidence_old", "evidence_new"]
    return changed[columns].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path, help="previous out/ folder")
    parser.add_argument("new", type=Path, help="new out/ folder")
    parser.add_argument("--out", type=Path, default=Path("out_compare.csv"))
    args = parser.parse_args()
    changes = compare_outputs(args.old, args.new)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    changes.to_csv(args.out, index=False)
    print(f"Готово: {len(changes)} изменений; файл {args.out.resolve()}")


if __name__ == "__main__":
    main()
