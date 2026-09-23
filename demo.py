#!/usr/bin/env python3
"""Print a compact 5-minute demo script from generated output CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def first_row(df: pd.DataFrame, fallback: dict):
    return df.iloc[0].to_dict() if not df.empty else fallback


def build_demo(out: Path) -> str:
    nodes = pd.read_csv(out / "nodes_roles.csv")
    clusters = pd.read_csv(out / "clusters.csv")
    top = pd.read_csv(out / "top_nodes.csv")
    cycles = pd.read_csv(out / "cycles.csv")
    routes = pd.read_csv(out / "routes.csv")
    gaps = pd.read_csv(out / "data_gaps.csv")

    top1 = first_row(top, {"gid": "нет", "role": "нет", "why": "нет данных"})
    boundary = first_row(
        nodes[nodes.boundary].sort_values(["priority_score", "gid"], ascending=[False, True]),
        {"gid": "нет", "p_hidden_outgoing": 0, "evidence": "нет данных"},
    )
    cluster = first_row(
        clusters[clusters.n_seed >= 2].sort_values(["n_seed", "n_nodes"], ascending=[False, False]),
        first_row(clusters.sort_values("n_nodes", ascending=False), {"cluster_id": "нет", "hypothesis": "нет данных"}),
    )
    route = first_row(routes.sort_values(["relay_days", "forwarded_kzt"], ascending=[False, False]),
                      {"path": "нет", "kind": "нет", "forwarded_kzt": 0})
    cycle = first_row(cycles, {"path": "нет", "bottleneck_kzt": 0})
    gap = first_row(gaps.sort_values("n_nodes", ascending=False), {"gap": "нет", "next_request": "нет данных"})

    lines = [
        "Демо-сценарий на 5 минут",
        "",
        f"1. Запуск: показать `./run.sh` и папку `{out}` с обязательными CSV.",
        f"2. Топ-узел: {top1['gid']} ({top1['role']}). Объяснение: {top1['why']}",
        f"3. Граница обхода: {boundary['gid']}; скрытый выход ~{float(boundary['p_hidden_outgoing']):.0%}. {boundary['evidence']}",
        f"4. Кластер: {cluster['cluster_id']}. {cluster['hypothesis']}",
        f"5. Маршрут пересылки: {route['path']} ({route['kind']}), сумма ~{float(route['forwarded_kzt']):,.0f} KZT.",
        f"6. Возвратный поток: {cycle['path']}, bottleneck {float(cycle['bottleneck_kzt']):,.0f} KZT.",
        f"7. Следующий запрос данных: {gap['gap']} -> {gap['next_request']}",
        "8. Завершение: подчеркнуть, что роли и маршруты являются гипотезами для проверки.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()
    print(build_demo(args.out))


if __name__ == "__main__":
    main()
