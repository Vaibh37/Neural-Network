"""Validate Markdown math fences and diagram links using Python's standard library."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(ROOT.glob("*.md")) + sorted((ROOT / "notes").glob("**/*.md"))
FENCE = re.compile(r"^\s*((?:\x60{3,}|~{3,}))([A-Za-z0-9_+-]*)\s*$")
REL_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
INVALID = (
    (re.compile(r"\\\x60"), "Escaped Markdown backtick found"),
    (re.compile(r"\\operatorname\{"), "Unsupported GitHub math command: use \\mathrm{...}"),
    (re.compile(r"\\\("), "Use dollar-delimited GitHub inline math"),
    (re.compile(r"\\\)"), "Use dollar-delimited GitHub inline math"),
)


def check(path: Path) -> list[str]:
    body = path.read_text(encoding="utf-8")
    lines = body.splitlines()
    failures: list[str] = []
    for pattern, message in INVALID:
        for match in pattern.finditer(body):
            line = body.count("\n", 0, match.start()) + 1
            failures.append(f"{path.relative_to(ROOT)}:{line}: {message}")

    opened: tuple[str, int] | None = None
    for n, line in enumerate(lines, start=1):
        match = FENCE.fullmatch(line)
        if match:
            marker, _language = match.groups()
            if opened is None:
                opened = (marker, n)
            elif marker[0] == opened[0][0] and len(marker) >= len(opened[0]):
                opened = None
            else:
                failures.append(f"{path.relative_to(ROOT)}:{n}: Mismatched code fence")
        elif re.match(r"^\s*(?:\x60{3,}|~{3,})\s*(?:math|python|bash)\S", line):
            failures.append(f"{path.relative_to(ROOT)}:{n}: Malformed opening code fence")

    if opened is not None:
        failures.append(f"{path.relative_to(ROOT)}:{opened[1]}: Unclosed code fence")

    for match in REL_IMAGE.finditer(body):
        href = match.group(1).strip()
        if href.startswith(("https://", "http://", "data:")):
            continue
        if not (path.parent / href.split("#", 1)[0]).is_file():
            failures.append(f"{path.relative_to(ROOT)}: Missing image: {href}")
    return failures


def main() -> int:
    failures = [error for path in FILES for error in check(path)]
    if failures:
        print("\n".join(failures))
        print(f"Markdown validation failed: {len(failures)} problem(s)")
        return 1
    print(f"Markdown format checks passed for {len(FILES)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
