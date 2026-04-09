"""
remove_demo.py

Remove one or more demos (by index) from a robomimic-style HDF5 file and
reorder the remaining demos so they are consecutively named demo_0, demo_1, ...

The script operates in-place: it writes a temporary file next to the source,
then atomically replaces the original (unless --out is given).

Usage examples
--------------
# Remove demo at index 5 (in-place)
python remove_demo.py --input dataset.hdf5 --indices 5

# Remove demos 0, 3, and 7 and write to a new file
python remove_demo.py --input dataset.hdf5 --indices 0 3 7 --out cleaned.hdf5

# Dry-run: just show which demos would be removed / kept
python remove_demo.py --input dataset.hdf5 --indices 2 --dry-run
"""

import argparse
import os
import shutil
import sys
import tempfile
from typing import List, Optional

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sorted_demo_keys(group: h5py.Group):
    """Return demo keys sorted numerically (demo_0, demo_1, ...), ignoring non-demo groups."""
    def _is_demo(name: str):
        parts = name.rsplit("_", 1)
        return len(parts) == 2 and parts[0] == "demo" and parts[1].isdigit()
    def _key(name: str):
        return int(name.rsplit("_", 1)[1])
    return sorted((k for k in group.keys() if _is_demo(k)), key=_key)


def copy_attrs(src, dst):
    """Copy all HDF5 attributes from src to dst."""
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def recompute_total(data_grp: h5py.Group) -> int:
    """Sum num_samples across all demos in data_grp."""
    total = 0
    for dk in sorted_demo_keys(data_grp):
        demo = data_grp[dk]
        if "num_samples" in demo.attrs:
            total += int(demo.attrs["num_samples"])
        elif "actions" in demo:
            total += demo["actions"].shape[0]
    return total


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def remove_demos(input_path: str, indices_to_remove: List[int],
                 output_path: Optional[str] = None, dry_run: bool = False):

    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with h5py.File(input_path, "r") as f_in:
        if "data" not in f_in:
            print("[ERROR] No 'data' group found in the HDF5 file.", file=sys.stderr)
            sys.exit(1)

        g_data = f_in["data"]
        all_demos = sorted_demo_keys(g_data)
        n_demos = len(all_demos)

        # ---- validate indices ----
        bad = [i for i in indices_to_remove if i < 0 or i >= n_demos]
        if bad:
            print(f"[ERROR] Index/indices out of range [0, {n_demos - 1}]: {bad}",
                  file=sys.stderr)
            sys.exit(1)

        remove_set = set(indices_to_remove)
        kept_demos  = [d for i, d in enumerate(all_demos) if i not in remove_set]
        removed_demos = [d for i, d in enumerate(all_demos) if i in remove_set]

        print(f"Total demos   : {n_demos}")
        print(f"To remove     : {removed_demos}  (indices {sorted(remove_set)})")
        print(f"Remaining     : {len(kept_demos)}")

        if dry_run:
            print("\n[DRY-RUN] No changes written.")
            return

        if not kept_demos:
            print("[ERROR] Removing all demos would leave an empty file. Aborting.",
                  file=sys.stderr)
            sys.exit(1)

        # ---- decide where to write ----
        in_place = (output_path is None)
        if in_place:
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix=".hdf5",
                dir=os.path.dirname(os.path.abspath(input_path)),
            )
            os.close(tmp_fd)
        else:
            tmp_path = output_path

        try:
            with h5py.File(tmp_path, "w") as f_out:
                # Copy top-level attributes
                copy_attrs(f_in, f_out)

                # Create data group and copy its attributes
                g_data_out = f_out.create_group("data")
                copy_attrs(g_data, g_data_out)

                # Copy kept demos with new consecutive names
                for new_idx, old_name in enumerate(kept_demos):
                    new_name = f"demo_{new_idx}"
                    f_out.copy(g_data[old_name], f"data/{new_name}")
                    print(f"  {old_name}  ->  {new_name}")

                # Update total samples
                new_total = recompute_total(g_data_out)
                g_data_out.attrs["total"] = new_total

                # Copy every top-level group/dataset except 'data' (e.g. 'mask')
                for key in f_in.keys():
                    if key == "data":
                        continue
                    f_out.copy(f_in[key], key)

                    # If this is a mask group, update the demo name strings
                    if key == "mask":
                        _update_mask(f_in["mask"], f_out["mask"],
                                     kept_demos, len(kept_demos))

            # Atomically replace original if in-place
            if in_place:
                shutil.move(tmp_path, input_path)
                print(f"\nFile updated in-place: {input_path}")
            else:
                print(f"\nOutput written to: {output_path}")

            print(f"Demos remaining : {len(kept_demos)}")
            print(f"Total samples   : {new_total}")

        except Exception:
            # Clean up temp file on failure
            if in_place and os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


def _update_mask(mask_in: h5py.Group, mask_out: h5py.Group,
                 kept_demos: List[str], n_kept: int):
    """
    Re-map demo names inside the mask group (train/valid splits).

    Old demo names that were removed are dropped; the remaining ones are
    translated to their new consecutive names.
    """
    # Build old-name -> new-name mapping
    name_map = {old: f"demo_{i}" for i, old in enumerate(kept_demos)}
    kept_set = set(kept_demos)

    dt = h5py.string_dtype(encoding="utf-8")

    for split in mask_in.keys():
        old_names = [n.decode() if isinstance(n, bytes) else n
                     for n in mask_in[split][:]]
        new_names = [name_map[n] for n in old_names if n in kept_set]

        # Delete existing dataset and recreate with updated names
        if split in mask_out:
            del mask_out[split]
        mask_out.create_dataset(split,
                                data=np.array(new_names, dtype=dt),
                                dtype=dt)
        print(f"  mask/{split}: {len(old_names)} -> {len(new_names)} entries")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Remove demo(s) by index from a robomimic HDF5 file and "
                    "reorder the remaining demo names consecutively."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input HDF5 file.",
    )
    parser.add_argument(
        "--indices", "-n", nargs="+", type=int, required=True,
        help="Zero-based index/indices of the demo(s) to remove. "
             "E.g. --indices 0 3 7",
    )
    parser.add_argument(
        "--out", "-o", default=None,
        help="Output HDF5 file path. If omitted the input file is updated "
             "in-place (via a temporary file for safety).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without writing anything.",
    )
    args = parser.parse_args()

    remove_demos(
        input_path=args.input,
        indices_to_remove=args.indices,
        output_path=args.out,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
