"""Run a prepared α,β-CROWN configuration and store structured result JSON.

Usage:
  # print command only
  python test.py --config configs/toy_linf_robustness.yaml

  # run verifier
  python test.py --config configs/toy_linf_robustness.yaml --run
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results" / "verification_result.json"


def _find_abcrown_entry(abcrown_root: Path) -> Optional[Path]:
    candidates = [
        abcrown_root / "complete_verifier" / "abcrown.py",
        abcrown_root / "abcrown.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _detect_status(text: str) -> str:
    lowered = text.lower()
    if "verified" in lowered or "unsafe" in lowered:
        if "unsat" in lowered and "sat" not in lowered:
            return "verified"
    if "unsafe" in lowered or re.search(r"\b(falsified|sat)\b", lowered):
        return "falsified"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "unknown" in lowered:
        return "unknown"
    return "finished"


def run_abcrown(config_path: Path, abcrown_root: Path, run_tool: bool, timeout: int = 120) -> Dict[str, Any]:
    config_path = config_path.resolve()
    abcrown_root = abcrown_root.resolve()
    entry = _find_abcrown_entry(abcrown_root)

    entry_exists = entry is not None
    command = None
    if entry is not None:
        command = f"python {entry} --config {config_path}"

    payload = {
        "config": str(config_path),
        "abcrown_root": str(abcrown_root),
        "command": command,
        "abcrown_entry_found": entry_exists,
        "status": None,
        "returncode": None,
        "runtime_seconds": None,
        "stdout_tail": "",
        "stderr_tail": "",
    }

    if not entry_exists:
        payload["status"] = "environment_not_ready"
        payload["stderr_tail"] = "abcrown.py not found. Clone and install alpha-beta-CROWN first."
        return payload

    if not run_tool:
        payload["status"] = "dry_run"
        return payload

    cmd: List[str] = ["python", str(entry), "--config", str(config_path)]
    started = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(abcrown_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        payload.update(
            {
                "status": "timeout",
                "returncode": None,
                "runtime_seconds": round(elapsed, 4),
                "stdout_tail": (exc.stdout or "")[:4000],
                "stderr_tail": (exc.stderr or "")[:4000],
            }
        )
        return payload

    elapsed = time.perf_counter() - started
    out = result.stdout or ""
    err = result.stderr or ""
    payload.update(
        {
            "returncode": int(result.returncode),
            "runtime_seconds": round(elapsed, 4),
            "stdout_tail": out[-8000:],
            "stderr_tail": err[-8000:],
            "status": _detect_status((out + "\n" + err)),
        }
    )
    return payload


def summarize_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/toy_linf_robustness.yaml"))
    parser.add_argument("--abcrown-root", type=Path, default=Path("."))
    parser.add_argument("--run", action="store_true", help="Run α,β-CROWN instead of dry run.")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")

    cfg = summarize_config(args.config)
    result = run_abcrown(
        config_path=args.config,
        abcrown_root=args.abcrown_root,
        run_tool=args.run,
        timeout=args.timeout,
    )
    result["config_preview"] = {
        "model": cfg.get("model", {}).get("name") if isinstance(cfg, dict) else None,
        "specification": cfg.get("specification") if isinstance(cfg, dict) else None,
        "data": cfg.get("data") if isinstance(cfg, dict) else None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print()
    if args.run:
        print("Result saved to", args.output)
    else:
        print("Dry run complete. Add --run to execute α,β-CROWN.")


if __name__ == "__main__":
    main()
