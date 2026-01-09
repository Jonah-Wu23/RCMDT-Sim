#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_table_md.py
=================
Build table.md by consolidating key experiment CSV/JSON outputs.
"""

import json
from pathlib import Path

import pandas as pd


def to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = ["" if pd.isna(v) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    root = Path(".")

    table_specs = [
        ("必做1-消融", root / "data/calibration/ablation/ablation_results.csv"),
        ("必做2-阈值敏感性", root / "data/calibration/sensitivity/threshold_sensitivity_results.csv"),
        ("必做3-IES对比", root / "data/calibration/ies_comparison/ies_comparison_results.csv"),
        ("必做3-IES配置", root / "data/calibration/ies_comparison/ies_config_for_paper.json"),
        ("选做2-全时段Heatmap-Base", root / "data/calibration/temporal_heatmap/heatmap_base.csv"),
        ("选做2-全时段Heatmap-Full", root / "data/calibration/temporal_heatmap/heatmap_full.csv"),
        ("必做5-Tail-Loss Ablation", root / "data/calibration/tail_loss_ablation_fixed/tail_loss_ablation_results.csv"),
    ]

    out = root / "table.md"
    lines = ["# 数据表汇总", "", "所有实验表格统一汇总在此文件。", ""]

    for title, path in table_specs:
        if not path.exists():
            lines.append(f"## {title}\n\n[缺失] {path.as_posix()}\n")
            continue

        lines.append(f"## {title}\n")
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame([data])
        else:
            df = pd.read_csv(path)

        if df.empty:
            lines.append("(空表)\n")
            continue

        lines.append(to_markdown(df))
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
