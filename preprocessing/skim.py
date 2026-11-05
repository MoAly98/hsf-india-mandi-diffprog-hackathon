"""Parallel skim of CMS Open Data NanoAOD.

Fans out across files with a process pool (one output file per input
file, so runs are resumable) and reads chunks concurrently with a
thread pool inside each process. Transient XRootD failures are retried.
The selection, branches, and per-event output are the same as
apply-preselection.py, which remains the simple single-process path.

Usage, driven by the datasets.json manifest:

    python skim.py --manifest datasets.json --sample all --output-dir skims

or on an explicit input like apply-preselection.py:

    python skim.py --input "root://.../*.root" --output-dir skims --is-mc
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import numpy as np

# scikit-hep
import awkward as ak
import fsspec
import uproot

TREE_NAME = "Events"
CUT = "HLT_TkMu50*(PuppiMET_pt>50)"
CHUNK_SIZE = 100_000
RETRIES = 3

BRANCHES = [
    "Muon_pt",
    "Muon_eta",
    "Muon_phi",
    "Muon_mass",
    "Muon_miniIsoId",
    "Muon_tightId",
    "FatJet_particleNet_TvsQCD",
    "FatJet_pt",
    "FatJet_eta",
    "FatJet_phi",
    "FatJet_mass",
    "Jet_btagDeepB",
    "Jet_jetId",
    "Jet_pt",
    "Jet_eta",
    "Jet_phi",
    "Jet_mass",
    "PuppiMET_pt",
    "PuppiMET_phi",
    "event",
    "run",
    "luminosityBlock",
]

MC_BRANCHES = ["genWeight", "Pileup_nTrueInt"]


def expand_files(pattern):
    """Expand a path or glob pattern (local or root://) into a file list."""
    if any(ch in pattern for ch in "*?["):
        fs, _, paths = fsspec.core.get_fs_token_paths(pattern)
        return [fs.unstrip_protocol(p) for p in sorted(paths)]
    return [pattern]


def _with_retry(fn, what):
    for attempt in range(RETRIES):
        try:
            return fn()
        except Exception as exc:
            if attempt == RETRIES - 1:
                raise
            wait = 2**attempt
            print(f"retry {attempt + 1} for {what} after error: {exc}", file=sys.stderr)
            time.sleep(wait)


def skim_file(input_path, output_path, is_mc, chunk_workers):
    """Skim one file; returns (input_path, nevts_in, nevts_out, seconds)."""
    t0 = time.perf_counter()
    branches = BRANCHES + (MC_BRANCHES if is_mc else [])
    tree = _with_retry(lambda: uproot.open(input_path)[TREE_NAME], input_path)
    n_entries = tree.num_entries

    bounds = [
        (start, min(start + CHUNK_SIZE, n_entries))
        for start in range(0, n_entries, CHUNK_SIZE)
    ]

    def read_chunk(start, stop):
        return _with_retry(
            lambda: tree.arrays(
                branches, cut=CUT, entry_start=start, entry_stop=stop
            ),
            f"{input_path}[{start}:{stop}]",
        )

    output = None
    output_tree = None
    branch_types = {}
    n_out = 0

    with ThreadPoolExecutor(max_workers=chunk_workers) as pool:
        futures = [pool.submit(read_chunk, *b) for b in bounds]
        for future in futures:
            arrays = future.result()

            if output is None:
                output = uproot.recreate(output_path)
                for branch in arrays.fields:
                    if isinstance(arrays[branch], ak.Array):
                        branch_types[branch] = arrays[branch].type
                    else:
                        branch_types[branch] = np.dtype(arrays[branch].dtype)
                output_tree = output.mktree(TREE_NAME, branch_types)

            if len(arrays) == 0:
                continue
            output_tree.extend({b: arrays[b] for b in branch_types})
            n_out += len(arrays)

    if output is not None:
        output.close()
    return input_path, n_entries, n_out, time.perf_counter() - t0


def build_tasks(args):
    """Yield (sample, input_path, output_path, is_mc, nevts) per file.

    ``nevts`` is the manifest's expected event count, None when unknown
    (``--input`` mode); it drives the progress bar total only.
    """
    out_root = Path(args.output_dir)
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        samples = list(manifest) if args.sample == "all" else [args.sample]
        for sample in samples:
            entry = manifest[sample]
            for item in entry["files"]:
                out = out_root / sample / Path(item["path"]).name
                yield sample, item["path"], out, bool(entry["is_mc"]), item.get("nevts")
    else:
        for path in expand_files(args.input):
            yield "adhoc", path, out_root / Path(path).name, args.is_mc, None


def main(args):
    tasks = []
    skipped = 0
    for sample, input_path, output_path, is_mc, nevts in build_tasks(args):
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tasks.append((sample, input_path, str(output_path), is_mc, nevts))

    print(f"{len(tasks)} file(s) to skim, {skipped} already done")
    if not tasks:
        return 0

    counts = [t[4] for t in tasks]
    total_events = sum(counts) if all(n is not None for n in counts) else None

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "skim_log.jsonl"

    failures = []
    totals = [0, 0]
    per_sample = {}
    t_start = time.perf_counter()
    with (
        ProcessPoolExecutor(max_workers=args.file_workers) as pool,
        log_path.open("a") as log,
        tqdm(total=total_events, unit="evt", unit_scale=True, desc="Skimming") as pbar,
    ):
        futures = {
            pool.submit(skim_file, inp, out, is_mc, args.workers): (sample, inp, nevts)
            for sample, inp, out, is_mc, nevts in tasks
        }
        for future in as_completed(futures):
            sample, inp, nevts = futures[future]
            record = {
                "sample": sample,
                "input": inp,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            try:
                _, n_in, n_out, seconds = future.result()
            except Exception as exc:
                failures.append((inp, exc))
                record["error"] = str(exc)
                pbar.update(nevts or 0)
                tqdm.write(f"FAILED {inp}: {exc}")
            else:
                record.update(nevts_in=n_in, nevts_out=n_out, seconds=round(seconds, 2))
                totals[0] += n_in
                totals[1] += n_out
                agg = per_sample.setdefault(sample, [0, 0, 0.0, 0])
                agg[0] += n_in
                agg[1] += n_out
                agg[2] += seconds
                agg[3] += 1
                pbar.update(n_in)
                tqdm.write(
                    f"{sample}/{Path(inp).name}: {n_in} -> {n_out} in {seconds:.1f}s"
                )
            log.write(json.dumps(record) + "\n")
            log.flush()

    wall = time.perf_counter() - t_start
    rate = totals[0] / wall if wall > 0 else 0.0
    print(
        f"done: {totals[0]} events in, {totals[1]} out, {len(failures)} failure(s) "
        f"in {wall / 60:.1f} min ({rate:,.0f} evt/s)"
    )
    for sample, (n_in, n_out, seconds, n_files) in per_sample.items():
        print(
            f"  {sample:15s} files={n_files:>3}  in={n_in:>11,}  out={n_out:>9,}  "
            f"file-time={seconds / 60:.1f} min"
        )
    print(f"per-file log: {log_path}")
    for inp, exc in failures:
        print(f"  failed: {inp}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argparser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="JSON manifest mapping sample -> {is_mc, files}",
    )
    argparser.add_argument(
        "--sample",
        type=str,
        default="all",
        help="Sample name from the manifest, or 'all'",
    )
    argparser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input ROOT file or glob pattern (alternative to --manifest)",
    )
    argparser.add_argument(
        "--is-mc",
        action="store_true",
        help="With --input: treat files as MC (adds genWeight branches)",
    )
    argparser.add_argument(
        "--output-dir",
        type=str,
        default="data/nanoaod",
        help="Directory for per-file skim outputs",
    )
    argparser.add_argument(
        "--file-workers",
        type=int,
        default=4,
        help="Concurrent files (processes)",
    )
    argparser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent chunk reads per file (threads)",
    )
    argparser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redo files whose output already exists",
    )
    args = argparser.parse_args()
    if bool(args.manifest) == bool(args.input):
        argparser.error("give exactly one of --manifest or --input")
    sys.exit(main(args))
