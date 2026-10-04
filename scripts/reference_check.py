#!/usr/bin/env python3
"""Check the face-wise product against an independent pure-Python reference.

The SLURM debug job only compares the parallel modes against the serial
mode. This script closes the loop: it writes random sparse input tensors
as CSV, runs the binary on them, and compares every mode against
C[:, :, s] = A[:, :, s] @ B[:, :, s] computed here without any of the
C++ code. Only the standard library is needed, so it runs anywhere
python3 is available.
"""

import argparse
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_tensors import compare  # noqa: E402


def random_sparse_tensor(rows, cols, depth, density, rng):
    tensor = defaultdict(dict)
    for s in range(depth):
        for r in range(rows):
            for c in range(cols):
                if rng.random() < density:
                    tensor[s][(r, c)] = rng.uniform(-1.0, 1.0)
    return tensor


def write_tensor(tensor, path):
    with path.open("w") as handle:
        for s in sorted(tensor):
            for (r, c), v in sorted(tensor[s].items()):
                handle.write(f"{s},{r},{c},{v:.17e}\n")


def facewise_product(a, b, depth):
    result = defaultdict(dict)
    for s in range(depth):
        b_rows = defaultdict(list)
        for (k, j), v in b[s].items():
            b_rows[k].append((j, v))
        acc = defaultdict(float)
        for (i, k), av in a[s].items():
            for j, bv in b_rows[k]:
                acc[(i, j)] += av * bv
        for pos, v in acc.items():
            if abs(v) > 1e-12:
                result[s][pos] = v
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", default="./tensor_app_cpu")
    parser.add_argument("--workdir", default="build/check")
    parser.add_argument("--modes", default="serial,omp,cuda,hybrid")
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # (rows_a, cols_a = rows_b, cols_b, depth, density): sparse COO path,
    # dense path (density >= 0.18 triggers the dense kernel), and a mix of
    # rectangular slices.
    cases = [
        (40, 30, 20, 5, 0.05),
        (24, 24, 24, 4, 0.60),
        (33, 17, 29, 6, 0.25),
    ]

    failures = 0
    for idx, (m, k, n, depth, density) in enumerate(cases):
        a = random_sparse_tensor(m, k, depth, density, rng)
        b = random_sparse_tensor(k, n, depth, density, rng)
        a_path = workdir / f"case{idx}_a.csv"
        b_path = workdir / f"case{idx}_b.csv"
        ref_path = workdir / f"case{idx}_ref.csv"
        write_tensor(a, a_path)
        write_tensor(b, b_path)
        write_tensor(facewise_product(a, b, depth), ref_path)

        for mode in args.modes.split(","):
            out_path = workdir / f"case{idx}_{mode}.csv"
            cmd = [
                args.bin, "--mode", mode, "--operation", "hadamard",
                "--rows-a", str(m), "--cols-a", str(k),
                "--rows-b", str(k), "--cols-b", str(n),
                "--depth", str(depth),
                "--input-a", str(a_path), "--input-b", str(b_path),
                "--warmup", "0", "--out", str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            label = f"case{idx} {m}x{k}x{depth} * {k}x{n}x{depth} d={density} mode={mode}"
            if proc.returncode != 0:
                print(f"[FAIL] {label}: binary exited {proc.returncode}: {proc.stderr.strip()}")
                failures += 1
                continue
            print(f"[{label}] ", end="")
            if compare(ref_path, out_path, args.tol) != 0:
                failures += 1

    print(f"reference check: {'FAILED' if failures else 'all passed'} ({failures} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
