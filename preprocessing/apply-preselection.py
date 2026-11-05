import argparse
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

import numpy as np

# scikit-hep
import awkward as ak
import fsspec
import uproot


def expand_files(pattern):
    """Expand a path or glob pattern (local or root://) into a file list."""
    if any(ch in pattern for ch in "*?["):
        fs, _, paths = fsspec.core.get_fs_token_paths(pattern)
        return [fs.unstrip_protocol(p) for p in sorted(paths)]
    return [pattern]


def process_with_selection(
    input_file,
    output_file,
    tree_name,
    branches,
    cut_str="",
    chunk_size=100_000,
    max_workers=8,
):
    """
    Process ROOT files by applying a selection on chunks of data and
    saving the filtered results to a new file, with a progress bar.

    Chunks are read and filtered concurrently (network + decompression
    overlap across threads) but written strictly in file/entry order,
    so the output is identical to a sequential pass.
    """

    files = expand_files(input_file)
    trees = [uproot.open(path)[tree_name] for path in files]

    # Entry counts come from the TTree headers already fetched by the
    # opens above; no event data is read for this.
    total_entries = sum(tree.num_entries for tree in trees)
    print(f"Found {total_entries} entries in {len(files)} file(s)")

    chunk_bounds = [
        (tree, start, min(start + chunk_size, tree.num_entries))
        for tree in trees
        for start in range(0, tree.num_entries, chunk_size)
    ]

    def read_chunk(tree, start, stop):
        return tree.arrays(
            branches, cut=cut_str, entry_start=start, entry_stop=stop
        )

    # Initialize output file and tree
    output = None
    output_tree = None
    branch_types = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(read_chunk, *bounds) for bounds in chunk_bounds]

        for future in tqdm(futures, desc="Processing events"):
            arrays = future.result()

            # For the first chunk, create the output file
            if output is None:
                output = uproot.recreate(output_file)

                # Remember the branch structure from the first successful chunk
                for branch in arrays.fields:
                    if isinstance(arrays[branch], ak.Array):
                        branch_types[branch] = arrays[branch].type
                    else:
                        branch_types[branch] = np.dtype(arrays[branch].dtype)

                # Create the output tree with proper types
                output_tree = output.mktree(tree_name, branch_types)

            # Make sure we only write available branches that match the output tree
            # This handles the case where some branches might be missing in later chunks
            available_branches = set(arrays.fields) & set(branch_types.keys())
            filtered_data_to_write = {
                branch: arrays[branch] for branch in available_branches
            }

            # Write the filtered data for available branches only
            output_tree.extend(filtered_data_to_write)

    # Close the output file if it was created
    if output is not None:
        output.close()
        print(f"Saved filtered data to {output_file}")
        return True
    else:
        print(f"No events passed selection for {input_file}")
        return False


def main(args):

    branches = [
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
        "PuppiMET_pt",
        "PuppiMET_phi",
        "event",
        "run",
        "luminosityBlock",
    ]

    if args.is_mc:
        branches += ["genWeight", "Pileup_nTrueInt"]

    process_with_selection(
        input_file=args.input,
        output_file=args.output,
        tree_name="Events",
        branches=branches,
        cut_str="HLT_TkMu50*(PuppiMET_pt>50)",
        chunk_size=100_000,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Apply preselection to ROOT files")
    argparser.add_argument(
        "--input",
        type=str,
        # default="root://eospublic.cern.ch//eos/opendata/cms/Run2016H/SingleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/280000/F302A865-17B0-064B-8154-41526BB38244.root",
        default="root://eospublic.cern.ch//eos/opendata/cms/Run2016H/SingleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/*/*.root",
        help="Input ROOT file to process (can be single file or glob pattern)",
    )
    argparser.add_argument(
        "--output",
        type=str,
        # default="/eos/user/l/ligerlac/z-prime-ttbar-data/filtered_data.root",
        default="filtered_data.root",
        help="Output file to save the filtered data",
    )
    argparser.add_argument(
        "--branch-file",
        type=str,
        default="branches.txt",
        help="File containing the branches to be used",
    )
    argparser.add_argument(
        "--is-mc",
        action="store_true",
        help="Flag to indicate if the input file is from MC (default: False)",
    )
    argparser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent chunk readers",
    )
    main(argparser.parse_args())
