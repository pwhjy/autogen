"""Remove macOS AppleDouble sidecar files from experiment paths.

The external SSD used for VACTHBench experiments can create ``._*`` files.
Those files confuse Git pack indexes, uv wheel installs, Docker build contexts,
and result scanners. This helper is intentionally narrow: it only targets
AppleDouble-style sidecars and prints every removal unless ``--quiet`` is used.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_DIR.parents[4]


DEFAULT_TARGETS = (
    REPO_ROOT / ".git" / "objects" / "pack",
    REPO_ROOT / "python" / ".venv",
    REPO_ROOT / "python" / "packages",
    Path.home() / ".cache" / "uv",
    BENCHMARK_DIR / "Tasks",
    BENCHMARK_DIR / "Results",
)
DEFAULT_PRUNE_DIRS = {
    ".agbench_venv",
    ".git",
    "__pycache__",
    "data",
    "node_modules",
    "workspace",
}


@dataclass
class CleanStats:
    files: int = 0
    dirs: int = 0
    errors: int = 0

    @property
    def total(self) -> int:
        return self.files + self.dirs


def is_sidecar(path: Path) -> bool:
    return path.name.startswith("._") or path.name == "__MACOSX"


def clean_path(root: Path, *, delete: bool, quiet: bool, prune_dirs: set[str] | None = None) -> CleanStats:
    stats = CleanStats()
    prune_dirs = DEFAULT_PRUNE_DIRS if prune_dirs is None else prune_dirs
    if not root.exists():
        if not quiet:
            print(f"missing {root}")
        return stats

    # rglob() can enter directories that are then deleted, so use an explicit
    # stack and skip descendants after deleting a sidecar directory.
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir()) if current.is_dir() else []
        except OSError as exc:
            stats.errors += 1
            if not quiet:
                print(f"error listing {current}: {exc}")
            continue

        for child in children:
            if is_sidecar(child):
                kind = "dir" if child.is_dir() else "file"
                if not quiet:
                    action = "remove" if delete else "would remove"
                    print(f"{action} {kind} {child}")
                if delete:
                    try:
                        if child.is_dir():
                            shutil.rmtree(child)
                            stats.dirs += 1
                        else:
                            child.unlink()
                            stats.files += 1
                    except OSError as exc:
                        stats.errors += 1
                        if not quiet:
                            print(f"error removing {child}: {exc}")
                else:
                    if child.is_dir():
                        stats.dirs += 1
                    else:
                        stats.files += 1
                continue
            if child.is_dir() and child.name not in prune_dirs:
                stack.append(child)
    return stats


def iter_workspace_git_pack_dirs(root: Path) -> list[Path]:
    """Find result worktree Git pack dirs without descending into workspaces."""
    pack_dirs: list[Path] = []
    if not root.exists() or not root.is_dir():
        return pack_dirs

    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            if child.name == "workspace":
                pack_dir = child / ".git" / "objects" / "pack"
                if pack_dir.is_dir():
                    pack_dirs.append(pack_dir)
                continue
            if child.name in DEFAULT_PRUNE_DIRS:
                continue
            stack.append(child)
    return pack_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=[str(path) for path in DEFAULT_TARGETS],
        help="Paths to scan. Defaults to the repo Git pack dir, python/.venv, VACTHBench Tasks, and Results.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually remove sidecars. Without this flag, the command is a dry run.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    total = CleanStats()
    for value in args.paths:
        root = Path(value).expanduser().resolve()
        stats = clean_path(root, delete=args.delete, quiet=args.quiet)
        total.files += stats.files
        total.dirs += stats.dirs
        total.errors += stats.errors
        for pack_dir in iter_workspace_git_pack_dirs(root):
            stats = clean_path(pack_dir, delete=args.delete, quiet=args.quiet, prune_dirs=set())
            total.files += stats.files
            total.dirs += stats.dirs
            total.errors += stats.errors

    action = "removed" if args.delete else "would remove"
    print(f"{action}: files={total.files} dirs={total.dirs} errors={total.errors}")
    if total.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
