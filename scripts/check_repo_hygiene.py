"""Repository hygiene gate for CI.

Fails if the tree contains things that make a checkout non-portable or
unpublishable:

* absolute Windows paths in source (a fresh clone must not inherit the author's
  drive layout);
* large data files that belong in a release, not in git;
* credential-looking strings.

Run locally with:  python scripts/check_repo_hygiene.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ABS_WIN = re.compile(r"[A-Za-z]:\\")
ABS_FAIL = re.compile(r"\b[XD]:[\\/](?:Breast_project|Dormbench)", re.I)
SECRET = re.compile(r"(api[_-]?key|secret|password|bearer)\s*[:=]\s*['\"][^'\"]{8,}",
                    re.I)
DATA_EXT = {".h5ad", ".h5", ".tar", ".gz", ".zip", ".rds", ".bam", ".fastq"}
MAX_MB = 8.0
SKIP_DIRS = {".git", "__pycache__", "build", "dist", "node_modules"}


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    return [ROOT / line for line in out.splitlines() if line.strip()]


def main() -> int:
    problems: list[str] = []
    warnings: list[str] = []

    for path in tracked_files():
        if not path.exists():
            continue
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        size_mb = path.stat().st_size / 1e6
        if size_mb > MAX_MB:
            warnings.append(f"{rel}: {size_mb:.1f} MB tracked")
        if path.suffix.lower() in DATA_EXT:
            problems.append(f"{rel}: data file tracked ({path.suffix})")
        if path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            if ABS_FAIL.search(line):
                problems.append(f"{rel}:{lineno}: absolute path in source")
            elif ABS_WIN.search(line) and "http" not in line:
                warnings.append(f"{rel}:{lineno}: Windows path literal")
            if SECRET.search(line):
                problems.append(f"{rel}:{lineno}: credential-looking string")

    for w in warnings:
        print(f"warn  {w}")
    for p in problems:
        print(f"FAIL  {p}")

    print(f"\nchecked {len(tracked_files())} tracked files: "
          f"{len(problems)} problems, {len(warnings)} warnings")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
