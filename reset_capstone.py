# reset_capstone.py
# Clears selected generated capstone outputs while preserving the project structure.

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def project_root() -> Path:
    """
    Resolve the project root.

    This script is expected to live in:
        project_tools/reset_capstone.py

    Therefore, the project root is one directory above project_tools.
    """
    try:
        script_path = Path(__file__).resolve()
        if script_path.parent.name == "project_tools":
            return script_path.parent.parent
        return script_path.parent
    except NameError:
        return Path.cwd().resolve()


def is_safe_target(root: Path, target: Path) -> bool:
    """
    Prevent accidental deletion outside the project root or deletion of the root itself.
    """
    root = root.resolve()
    target = target.resolve()

    if target == root:
        return False

    try:
        target.relative_to(root)
    except ValueError:
        return False

    return True


def clear_folder(folder: Path, *, dry_run: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    for item in folder.iterdir():
        if dry_run:
            print(f"[dry-run] Would remove: {item}")
            continue

        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clear generated capstone outputs while keeping required folders."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files. Without this flag, the script only previews actions.",
    )
    args = parser.parse_args()

    root = project_root()
    dry_run = not args.execute

    targets = [
        root / "artifacts",
        root / "logs",
        root / "truths",
        root / "data" / "bronze" / "train",
        root / "data" / "silver" / "train",
        root / "data" / "gold" / "train",
    ]

    print(f"Project root: {root}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")

    for folder in targets:
        if not is_safe_target(root, folder):
            raise RuntimeError(f"Unsafe deletion target blocked: {folder}")

        print(f"Clearing: {folder}")
        clear_folder(folder, dry_run=dry_run)

    print("Done.")


if __name__ == "__main__":
    main()