@'
# reset_capstone.py
# Deletes contents of selected folders (keeps the folders).

from __future__ import annotations

from pathlib import Path
import shutil


def project_root() -> Path:
    # Project root = folder where this script lives (best practice if saved in project root)
    try:
        return Path(__file__).resolve().parent
    except NameError:
        # Fallback for interactive execution
        return Path.cwd().resolve()


def clear_folder(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for item in folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def main() -> None:
    root = project_root()

    targets = [
        root / "artifacts",
        root / "logs",
        root / "data" / "bronze" / "train",
        root / "data" / "silver" / "train",
    ]

    for folder in targets:
        print(f"Clearing: {folder}")
        clear_folder(folder)

    print("Done.")


if __name__ == "__main__":
    main()
'@ | Set-Content -Encoding UTF8 -Path .\reset_capstone.py