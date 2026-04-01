#!/usr/bin/env python3
"""Thin wrapper that forces strict bridge-side `phystwin` mode."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import warp as wp


BRIDGE_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
DEMOS_DIR = BRIDGE_ROOT / "demos"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if str(DEMOS_DIR) not in sys.path:
    sys.path.insert(0, str(DEMOS_DIR))

from bridge_bootstrap import ensure_bridge_runtime_paths  # noqa: E402

ensure_bridge_runtime_paths()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


newton_import_ir = _load_module(
    "bridge_phystwin_self_collision_newton_import_ir",
    CORE_DIR / "newton_import_ir.py",
)


def main() -> int:
    args = newton_import_ir.parse_args()
    cfg = newton_import_ir.SimConfig.from_args(args)
    cfg.self_contact_mode = newton_import_ir.SELF_CONTACT_MODE_PHYSTWIN
    cfg.disable_particle_contact_kernel = True

    wp.init()
    ir = newton_import_ir.load_ir(cfg.ir_path)
    model_result = newton_import_ir.build_model(ir, cfg, cfg.device)
    sim_result = newton_import_ir.simulate(
        model_result.model,
        ir,
        cfg,
        cfg.device,
        model_result=model_result,
    )
    summary = newton_import_ir.save_results(cfg, ir, model_result, sim_result)

    json_path = cfg.out_dir / f"{cfg.output_prefix}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
