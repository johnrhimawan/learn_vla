"""Stamp evaluation reports with the commit that produced them."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def repository_state() -> dict[str, Any]:
    """Return the commit a report was produced from, and whether it was dirty.

    Every report-producing script embeds this so a result can be traced back
    to the exact source that made it.  ``tracked_files_dirty`` must be false
    for a report to count as evidence, which is why audits are run from a
    clean worktree.

    The repository root is resolved by git rather than by counting parent
    directories, so moving a module between packages cannot silently change
    which directory is inspected.
    """
    here = Path(__file__).resolve().parent

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=here,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "git_revision": git("rev-parse", "HEAD"),
        "tracked_files_dirty": bool(
            git("status", "--porcelain", "--untracked-files=no")
        ),
    }
