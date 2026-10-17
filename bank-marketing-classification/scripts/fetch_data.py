"""Re-download the dataset from the UCI repository into data/raw/.

Only needed if data/raw/ is missing; the CSVs are already committed.
"""
import io
import sys
import zipfile
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    # The archive nests two further zips (bank.zip, bank-additional.zip).
    with zipfile.ZipFile(io.BytesIO(resp.content)) as outer:
        for inner_name in outer.namelist():
            if not inner_name.endswith(".zip"):
                continue
            with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
                for member in inner.namelist():
                    name = Path(member).name
                    if name.endswith((".csv", ".txt")) and not name.startswith("."):
                        (RAW_DIR / name).write_bytes(inner.read(member))
                        print(f"wrote {RAW_DIR / name}")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        sys.exit(f"Download failed: {exc}")
