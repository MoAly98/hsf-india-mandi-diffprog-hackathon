"""Record pre-skim genWeight sums in the manifest.

Reads only the genWeight branch of every MC file in the manifest
(concurrently, with the same retry policy as the skim) and writes the
per-file sum back into the manifest as ``sum_genweight``. Data samples
(``is_mc: false``) are left untouched. Rerunning skips files that
already carry a sum unless ``--overwrite`` is given.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import uproot

from skim import TREE_NAME, _with_retry


def sum_genweight(path):
    def read():
        with uproot.open(path) as f:
            return float(f[TREE_NAME]["genWeight"].array(library="np").sum())

    return _with_retry(read, path)


def main(args):
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())

    todo = [
        item
        for sample in manifest.values()
        if sample["is_mc"]
        for item in sample["files"]
        if args.overwrite or "sum_genweight" not in item
    ]
    print(f"{len(todo)} MC file(s) to read")

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(sum_genweight, item["path"]): item for item in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc="genWeight"):
            item = futures[future]
            try:
                item["sum_genweight"] = future.result()
            except Exception as exc:
                failures.append((item["path"], exc))
                tqdm.write(f"FAILED {item['path']}: {exc}")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"manifest updated: {manifest_path}")
    for sample, entry in manifest.items():
        if not entry["is_mc"]:
            continue
        sums = [f.get("sum_genweight") for f in entry["files"]]
        done = [s for s in sums if s is not None]
        print(
            f"  {sample:15s} files={len(done)}/{len(sums)}  "
            f"sum_genweight={sum(done):.6g}"
        )
    for path, exc in failures:
        print(f"  failed: {path}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argparser.add_argument(
        "--manifest",
        type=str,
        default=str(Path(__file__).parent / "datasets.json"),
        help="JSON manifest to read and update in place",
    )
    argparser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Concurrent file reads",
    )
    argparser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute sums already present in the manifest",
    )
    raise SystemExit(main(argparser.parse_args()))
