#!/usr/bin/env python3
"""Verify the committed submission tree from a normal Git checkout.

verify_submission.py intentionally expects a freshly extracted release tree with
no .git directory or untracked files. This helper reconstructs exactly the files
listed in SHA256SUMS into a temporary directory, then runs the release verifier
there. Optionally it also checks a separately cloned upstream srs-benchmark
checkout against the source identity recorded by the full run.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def listed_files(root: Path) -> list[Path]:
    paths = []
    seen = set()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        _, relative = line.split("  ", 1)
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in seen:
            raise ValueError(f"unsafe or duplicate checksum path: {relative}")
        seen.add(relative)
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--upstream",
        type=Path,
        help="optional pinned sparse srs-benchmark checkout to verify as well",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    with tempfile.TemporaryDirectory(prefix="fsrs7-residualboost-verify-") as temp:
        stage = Path(temp) / "submission"
        stage.mkdir()
        shutil.copy2(root / "SHA256SUMS", stage / "SHA256SUMS")
        for relative in listed_files(root):
            source = root / relative
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"missing or symlinked listed file: {source}")
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        command = [
            sys.executable,
            str(stage / "evaluation/submission-20260919/verify_submission.py"),
            "--root",
            str(stage),
        ]
        if args.upstream is not None:
            command.extend(["--upstream", str(args.upstream.resolve())])
        subprocess.run(command, cwd=stage, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
