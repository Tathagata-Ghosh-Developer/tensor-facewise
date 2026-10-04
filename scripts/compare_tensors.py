#!/usr/bin/env python3

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import List


def load_tensor(path: Path):
    entries = defaultdict(dict)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            slice_index, row_index, col_index, value = row[:4]
            entries[int(slice_index)][(int(row_index), int(col_index))] = float(value)
    return entries


def values_close(left_val: float, right_val: float, abs_tol: float, rel_tol: float) -> bool:
    delta = abs(left_val - right_val)
    if delta <= abs_tol:
        return True
    scale = max(abs(left_val), abs(right_val))
    if scale == 0.0:
        return delta <= abs_tol
    return (delta / scale) <= rel_tol


def compare(left_path: Path, right_path: Path, tolerance: float) -> int:
    left = load_tensor(left_path)
    right = load_tensor(right_path)
    all_slices = sorted(set(left) | set(right))
    abs_tol = tolerance
    rel_tol = 1e-3
    max_abs_error = 0.0
    max_rel_error = 0.0
    mismatch_count = 0

    for slice_index in all_slices:
        left_slice = left.get(slice_index, {})
        right_slice = right.get(slice_index, {})
        all_positions = set(left_slice) | set(right_slice)
        for position in all_positions:
            left_val = left_slice.get(position, 0.0)
            right_val = right_slice.get(position, 0.0)
            delta = abs(left_val - right_val)
            max_abs_error = max(max_abs_error, delta)

            scale = max(abs(left_val), abs(right_val))
            rel_err = (delta / scale) if scale > 0.0 else 0.0
            max_rel_error = max(max_rel_error, rel_err)

            if not values_close(left_val, right_val, abs_tol, rel_tol):
                if mismatch_count < 5:
                    print(
                        f"Mismatch at slice {slice_index}, position {position}: "
                        f"{left_val:.6e} vs {right_val:.6e} "
                        f"(abs_err={delta:.6e}, rel_err={rel_err:.6e})"
                    )
                mismatch_count += 1

    if mismatch_count > 0:
        print(
            f"FAIL: {mismatch_count} mismatches "
            f"(abs_max={max_abs_error:.6e}, rel_max={max_rel_error:.6e}, "
            f"abs_tol={abs_tol:.1e}, rel_tol={rel_tol:.1e})"
        )
        return 1

    print(
        f"PASS abs_max={max_abs_error:.6e} rel_max={max_rel_error:.6e} "
        f"(abs_tol={abs_tol:.1e}, rel_tol={rel_tol:.1e})"
    )
    return 0


def main(argv: List[str]) -> int:
    if len(argv) != 4:
        print("Usage: compare_tensors.py left.csv right.csv tolerance", file=sys.stderr)
        return 2
    try:
        return compare(Path(argv[1]), Path(argv[2]), float(argv[3]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
