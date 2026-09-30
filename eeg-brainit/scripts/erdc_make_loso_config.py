#!/usr/bin/env python3
"""Write per-subject ATM distill Stage-1 / Stage-3 override YAMLs."""

from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from erdc_loso_paths import distill_root, resolve_stage_ckpt, subject_dir_suffix  # noqa: E402


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True, help="e.g. sub-08")
    parser.add_argument("--configs-dir", type=str, default="configs")
    args = parser.parse_args()

    sub = args.subject
    suffix = subject_dir_suffix(sub)
    cfg_dir = Path(args.configs_dir)
    if not cfg_dir.is_absolute():
        cfg_dir = ROOT / cfg_dir

    s1_base = load_yaml(ROOT / "configs/atm_distill_s1_sub08.yaml")
    s1_base["output_dir"] = f"outputs/atm_distill_s1_{suffix}"
    s1_base["atm"]["subject"] = sub
    s1_base["train"]["init_checkpoint"] = ""
    s1_base["train"]["min_distill_cos"] = 0.95
    s1_base["train"]["min_atm_top1_for_ckpt"] = 0.25

    s3_base = load_yaml(ROOT / "configs/atm_distill_s3_sub08.yaml")
    s3_base["output_dir"] = f"outputs/atm_distill_s3_{suffix}"
    s3_base["atm"]["subject"] = sub
    s3_base["train"]["min_distill_cos"] = 0.95
    s3_base["train"]["min_atm_top1_for_ckpt"] = 0.25

    s1_ckpt = resolve_stage_ckpt(ROOT, sub, 1)
    s3_base["train"]["init_checkpoint"] = (
        str(s1_ckpt.relative_to(ROOT)) if s1_ckpt is not None else ""
    )

    s1_path = cfg_dir / f"_loso_s1_{sub}.yaml"
    s3_path = cfg_dir / f"_loso_s3_{sub}.yaml"
    write_yaml(s1_path, s1_base)
    write_yaml(s3_path, s3_base)
    print(f"[OK] {s1_path}")
    print(f"[OK] {s3_path}")
    print(f"[INFO] output suffix={suffix}")
    print(f"[INFO] s3 init_checkpoint={s3_base['train']['init_checkpoint']}")


if __name__ == "__main__":
    main()
