"""Check the rank-local assets against the published reference before launch."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(path, row, size):
    if path.stat().st_size != row[size] or digest(path) != row["sha256"]:
        raise SystemExit(f"Unexpected contents of {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank", type=int, choices=[0, 1])
    args = parser.parse_args()
    ref = json.loads(Path(__file__).with_name("reference-weights.json").read_text())
    for name, row in ref["model_view"].items():
        check(Path("/model") / name, row, "bytes")
    matches = list(Path("/cache/rank-cache").glob(f"rank{args.rank}of2-*.bin"))
    if len(matches) != 1:
        raise SystemExit(f"Expected one cache for rank {args.rank}, found {len(matches)}")
    check(matches[0], ref["ranks"][args.rank], "total_bytes")
    print(f"Verified model view and rank {args.rank} cache")
