from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the Bag-of-Lies archive.")
    parser.add_argument("--zip", default="data/BagOfLies.zip", help="Path to BagOfLies.zip.")
    parser.add_argument("--out", default="data/raw", help="Extraction directory.")
    parser.add_argument("--password", default=None, help="Zip password. Omit to try without one.")
    parser.add_argument(
        "--members",
        nargs="*",
        default=None,
        help="Optional archive members to extract, e.g. BagOfLies/Annotations.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = Path(args.zip)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    password = args.password.encode("utf-8") if args.password else None

    with zipfile.ZipFile(archive) as zf:
        members = args.members if args.members else zf.namelist()
        zf.extractall(out_dir, members=members, pwd=password)

    print(f"Extracted {len(members)} entries to {out_dir}")


if __name__ == "__main__":
    main()
