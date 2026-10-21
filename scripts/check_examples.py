"""Run every educational example in a separate Python process.

This catches syntax errors, broken imports, and failed example assertions
without importing modules with top-level training loops into this process.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def main() -> int:
    scripts = sorted(EXAMPLES.glob("*.py"))
    if not scripts:
        print("No educational examples found", file=sys.stderr)
        return 1
    failed = []
    for script in scripts:
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            failed.append(f"{script.name}: exceeded 60 seconds")
            continue
        if result.returncode:
            failed.append(
                f"{script.name}: exit {result.returncode}\n"
                f"{result.stdout}\n{result.stderr}"
            )
        else:
            print(f"PASS {script.name}")
    if failed:
        print("\n".join(failed), file=sys.stderr)
        return 1
    print(f"All {len(scripts)} example scripts passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
