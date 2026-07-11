from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = PROJECT_ROOT / "data" / "kaggle"


DATASETS = [
    ("mlg-ulb/creditcardfraud", "creditcard.zip"),
    ("ealaxi/paysim1", "paysim1.zip"),
]


def main() -> int:
    if not shutil.which("kaggle"):
        print("Kaggle CLI is not installed.")
        print("Install it with: pip install kaggle")
        print("Then place kaggle.json in C:\\Users\\<you>\\.kaggle\\kaggle.json")
        return 1

    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, zip_name in DATASETS:
        print(f"Downloading {dataset}...")
        subprocess.check_call(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset,
                "-p",
                str(KAGGLE_DIR),
                "--force",
            ]
        )
        zip_path = KAGGLE_DIR / zip_name
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(KAGGLE_DIR)
            print(f"Extracted {zip_path.name}")

    print("Done. Restart backend/server.py so the model trains with Kaggle data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
