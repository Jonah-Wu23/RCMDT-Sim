#!/usr/bin/env python
"""Run immutable stages of the SMC camera-ready experiment protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_experiments.pipeline import (  # noqa: E402
    capture_environment,
    load_protocol_manifest,
    run_pilot,
    verify_input_hashes,
    write_json_immutable,
)
from src.paper_experiments.l1_stage import run_l1_stage  # noqa: E402
from src.paper_experiments.ablation_stage import run_ablation_stage  # noqa: E402
from src.paper_experiments.evaluation_stage import run_evaluation_stage  # noqa: E402


def prepare(sumo_binary: str) -> dict:
    manifest = load_protocol_manifest(PROJECT_ROOT)
    verified = verify_input_hashes(PROJECT_ROOT, manifest)
    output_root = PROJECT_ROOT / manifest["outputs"]["run_directory"]
    write_json_immutable(output_root / "manifests" / "effective_manifest.json", manifest)
    write_json_immutable(
        output_root / "manifests" / "input-verification.json",
        {"schema_version": "paper-input-verification/v1", "verified_sha256": verified},
    )
    environment = capture_environment(PROJECT_ROOT, sumo_binary=sumo_binary)
    write_json_immutable(output_root / "environment" / "environment.json", environment)
    return {"stage": "prepare", "verified_inputs": len(verified), "output_root": str(output_root)}


def run_l1(sumo_binary: str, workers: int) -> dict:
    manifest = load_protocol_manifest(PROJECT_ROOT)
    verify_input_hashes(PROJECT_ROOT, manifest)
    software = capture_environment(PROJECT_ROOT, sumo_binary=sumo_binary)["software_versions"]
    timeout_s = float(manifest["execution"]["post_pilot_timeout_s"])
    result = run_l1_stage(
        PROJECT_ROOT,
        manifest,
        software,
        sumo_binary,
        workers,
        timeout_s,
    )
    output_root = PROJECT_ROOT / manifest["outputs"]["run_directory"]
    write_json_immutable(output_root / "l1" / "stage-result.json", result)
    return {"stage": "l1", **result}


def _selected_l1_parameters(manifest: dict) -> dict[int, dict[str, float]]:
    output_root = PROJECT_ROOT / manifest["outputs"]["run_directory"]
    selected: dict[int, dict[str, float]] = {}
    for seed in manifest["ablation"]["seeds"]:
        path = output_root / "l1" / f"seed-{int(seed)}" / "selected.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = payload["selected_for_l2"]
            if payload["status"] != "succeeded" or value["method"] != "BO":
                raise ValueError("selected L1 record is not a successful BO result")
            selected[int(seed)] = {
                key: float(value["parameters"][key])
                for key in ("t_board", "t_fixed", "tau", "sigma", "minGap_bus", "accel", "decel")
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot load frozen L1 parameters for seed={seed}: {path}: {exc}") from exc
    return selected


def run_ablation(sumo_binary: str, workers: int) -> dict:
    manifest = load_protocol_manifest(PROJECT_ROOT)
    verify_input_hashes(PROJECT_ROOT, manifest)
    timeout_s = float(manifest["execution"]["post_pilot_timeout_s"])
    result = run_ablation_stage(
        PROJECT_ROOT,
        selected_l1_by_seed=_selected_l1_parameters(manifest),
        workers=workers,
        timeout=timeout_s,
        sumo_binary=sumo_binary,
        base_manifest=manifest,
    )
    return {"stage": "ablation", **result}


def run_evaluation() -> dict:
    manifest = load_protocol_manifest(PROJECT_ROOT)
    result = run_evaluation_stage(PROJECT_ROOT, base_manifest=manifest)
    output_root = PROJECT_ROOT / manifest["outputs"]["run_directory"]
    write_json_immutable(output_root / "evaluation" / "stage-result.json", result)
    return {"stage": "evaluate", **result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "pilot", "l1", "ablation", "evaluate"))
    parser.add_argument("--sumo-binary", default="sumo")
    parser.add_argument("--workers-for-estimate", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare(args.sumo_binary)
    elif args.stage == "pilot":
        _, result = run_pilot(
            PROJECT_ROOT,
            sumo_binary=args.sumo_binary,
            workers_for_estimate=args.workers_for_estimate,
        )
    elif args.stage == "l1":
        result = run_l1(args.sumo_binary, args.workers)
    elif args.stage == "ablation":
        result = run_ablation(args.sumo_binary, args.workers)
    else:
        result = run_evaluation()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
