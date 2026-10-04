#!/usr/bin/env python3
# pyright: reportGeneralTypeIssues=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""Generate full charts and tables for tensor experiment artifacts.

This script ingests CSV and log outputs from debug/scalability/profile runs,
computes derived metrics, and writes:
- PNG + PDF charts
- Detailed and summary tables
- Asset manifests for LaTeX integration
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


MODE_MAP = {
    "serial": "serial",
    "omp": "omp",
    "openmp": "omp",
    "cuda": "cuda",
    "hybrid": "hybrid",
    "0": "serial",
    "1": "omp",
    "2": "cuda",
    "3": "hybrid",
}

MODE_ORDER = ["serial", "omp", "cuda", "hybrid"]
MODE_COLORS = {
    "serial": "#4c6faf",
    "omp": "#2a9d8f",
    "cuda": "#f4a261",
    "hybrid": "#e76f51",
}

PLOT_DPI = 220


def normalize_mode(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    key = str(value).strip().lower()
    return MODE_MAP.get(key, key)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        # determinism files are header-less
        df = pd.read_csv(path, header=None, low_memory=False)
    return coerce_numeric_like(df)


def coerce_numeric_like(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_object_dtype(out[col]):
            continue
        s = out[col].astype(str).str.strip()
        s = s.str.replace(r"^\.(\d+)$", r"0.\1", regex=True)
        converted = pd.to_numeric(s, errors="coerce")
        if not converted.empty and converted.notna().mean() >= 0.80:
            out[col] = converted
    return out


def parse_kv_line(line: str) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, val = token.split("=", 1)
        payload[key.strip()] = val.strip().rstrip(",")
    return payload


def parse_filename_hints(path: Path) -> Dict[str, object]:
    stem = path.stem
    parts = stem.split("_")
    hints: Dict[str, object] = {
        "file_stem": stem,
        "shape_class_hint": "",
        "tensor_shape_hint": "",
        "density_combo_hint": "",
        "operation_hint": "",
        "mode_hint": "",
        "storage_format_hint": "",
        "threads_hint": np.nan,
        "case_hint": "",
    }

    if parts and parts[0] in {"square", "tall", "wide"}:
        hints["shape_class_hint"] = parts[0]
        if len(parts) > 1:
            hints["mode_hint"] = normalize_mode(parts[1])
        if len(parts) > 2 and re.fullmatch(r"\d+x\d+x\d+", parts[2]):
            hints["tensor_shape_hint"] = parts[2]
        for token in parts[3:]:
            if re.fullmatch(r"t\d+", token):
                hints["threads_hint"] = int(token[1:])
        return hints

    if len(parts) >= 2 and parts[0] in {"dense", "sparse"} and parts[1] in {"dense", "sparse"}:
        hints["density_combo_hint"] = f"{parts[0]}_{parts[1]}"
    for token in parts:
        t = token.lower()
        if t in {"hadamard", "qr", "svd"}:
            hints["operation_hint"] = t
        if t in {"serial", "omp", "openmp", "cuda", "hybrid"}:
            hints["mode_hint"] = normalize_mode(t)
        if t in {"coo", "ell", "auto"}:
            hints["storage_format_hint"] = t
        if re.fullmatch(r"t\d+", t):
            hints["threads_hint"] = int(t[1:])

    if parts and parts[0].startswith("case"):
        hints["case_hint"] = parts[0]
        if "ratio" in parts:
            idx = parts.index("ratio")
            if idx + 1 < len(parts):
                try:
                    hints["threads_hint"] = float(parts[idx + 1])
                except Exception:
                    pass

    return hints


def parse_operation_log(path: Path, root: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    row: Dict[str, object] = {
        "source_file": str(path.relative_to(root)).replace("\\", "/"),
        "source_group": path.parts[-3] if len(path.parts) >= 3 else "unknown",
        "status": "ok",
        "mode": "",
        "operation": "",
        "time_sec": np.nan,
    }
    row.update(parse_filename_hints(path))

    op_line = next((ln for ln in lines if "mode=" in ln and "operation=" in ln and "time_sec=" in ln), "")
    metrics_line = next((ln for ln in lines if ln.startswith("metrics ")), "")
    storage_line = next((ln for ln in lines if ln.startswith("storage ")), "")

    if op_line:
        op_payload = parse_kv_line(op_line)
        row["mode"] = normalize_mode(op_payload.get("mode", ""))
        row["operation"] = str(op_payload.get("operation", "")).lower()
        try:
            row["time_sec"] = float(op_payload.get("time_sec", "nan"))
        except Exception:
            row["time_sec"] = np.nan
    else:
        row["status"] = "no_metrics"

    if metrics_line:
        for k, v in parse_kv_line(metrics_line).items():
            if k == "metrics":
                continue
            row[k] = v

    if storage_line:
        for k, v in parse_kv_line(storage_line).items():
            if k == "storage":
                continue
            row[f"storage_{k}"] = v

    lowered = text.lower()
    if "task 0: killed" in lowered or "force terminated" in lowered:
        row["status"] = "oom_killed"
    elif "error" in lowered and not op_line:
        row["status"] = "error"

    return row


def parse_slurm_size_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    size_pattern = re.compile(
        r"\[SIZE\] class=(?P<shape_class>\w+), output=(?P<tensor_shape>\d+x\d+x\d+), "
        r"A=(?P<input_shape_a>\d+x\d+x\d+), B=(?P<input_shape_b>\d+x\d+x\d+), "
        r"storage_input=(?P<storage_size_input>[0-9.eE+-]+), sparsity=(?P<sparsity>[0-9.eE+-]+), "
        r"input_mem=(?P<input_mem_mib>[0-9.eE+-]+)MiB, working_mem=(?P<working_set_mem_mib>[0-9.eE+-]+)MiB, "
        r"dense_equiv=(?P<dense_equiv_working_set_mem_mib>[0-9.eE+-]+)MiB"
    )

    records: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = size_pattern.search(line)
        if not match:
            continue
        rec = match.groupdict()
        rec["source_file"] = path.name
        records.append(rec)

    df = pd.DataFrame(records)
    return coerce_numeric_like(df)


def parse_slurm_events(paths: Iterable[Path]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            if not line.strip():
                continue
            severity = ""
            marker = ""
            low = line.lower()
            if "[error]" in low or "error:" in low:
                severity = "error"
            elif "[warn]" in low or "warning" in low:
                severity = "warn"

            if "killed" in low:
                marker = "oom_killed"
                severity = "error"
            elif "err_nvgpuctrperm" in low:
                marker = "permission"
                if not severity:
                    severity = "warn"
            elif "syntax error" in low:
                marker = "script_syntax"
                if not severity:
                    severity = "error"

            if not severity and not marker:
                continue

            rows.append(
                {
                    "source_file": path.name,
                    "line_no": line_no,
                    "severity": severity or "info",
                    "marker": marker,
                    "message": line.strip(),
                }
            )

    return pd.DataFrame(rows)


def shape_dim_from_string(shape: object) -> Tuple[float, float, float]:
    if shape is None or (isinstance(shape, float) and math.isnan(shape)):
        return np.nan, np.nan, np.nan
    text = str(shape)
    parts = text.split("x")
    if len(parts) != 3:
        return np.nan, np.nan, np.nan
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        return np.nan, np.nan, np.nan


def enrich_scalability(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["mode"] = out["mode"].map(normalize_mode)
    out["efficiency"] = np.where(out["threads"] > 0, out["speedup_vs_serial"] / out["threads"], np.nan)
    out["oversubscribed"] = out["threads"] > 36

    dims = out["tensor_shape"].map(shape_dim_from_string)
    out["rows_out"] = dims.map(lambda x: x[0])
    out["cols_out"] = dims.map(lambda x: x[1])
    out["depth"] = dims.map(lambda x: x[2])
    out["tensor_elements"] = out["rows_out"] * out["cols_out"] * out["depth"]
    out["memory_compression_dense_to_active"] = out.get("input_compression_dense_to_active", np.nan)
    out["memory_ratio_working_vs_dense"] = np.where(
        out["dense_equiv_working_set_mem_mib"] > 0,
        out["working_set_mem_mib"] / out["dense_equiv_working_set_mem_mib"],
        np.nan,
    )
    return out


def enrich_correctness(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["mode"] = out["mode"].map(normalize_mode)

    keys = ["density_combo", "operation", "storage_format", "tensor_shape"]
    baseline = (
        out[out["mode"] == "serial"][keys + ["time_sec"]]
        .rename(columns={"time_sec": "serial_time_sec"})
        .drop_duplicates(subset=keys)
    )
    out = out.merge(baseline, on=keys, how="left")
    out["speedup_vs_serial"] = np.where(
        (out["time_sec"] > 0) & (out["serial_time_sec"] > 0),
        out["serial_time_sec"] / out["time_sec"],
        np.nan,
    )
    out.loc[out["mode"] == "serial", "speedup_vs_serial"] = 1.0
    out["memory_ratio_working_vs_dense"] = np.where(
        out["dense_equiv_working_set_mem_mib"] > 0,
        out["working_set_mem_mib"] / out["dense_equiv_working_set_mem_mib"],
        np.nan,
    )
    return out


def enrich_bottleneck(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["mode"] = out["mode"].map(normalize_mode)

    def parse_thread_or_ratio(val: object) -> float:
        try:
            return float(val)
        except Exception:
            return np.nan

    out["threads_or_ratio_num"] = out["threads_or_ratio"].map(parse_thread_or_ratio)

    # case-local speedup baseline
    out["case_speedup"] = np.nan
    for case_name, chunk in out.groupby("case"):
        baseline_candidates = chunk[chunk["threads_or_ratio_num"] == chunk["threads_or_ratio_num"].min()]
        if baseline_candidates.empty:
            continue
        baseline_time = baseline_candidates["time_sec"].iloc[0]
        if baseline_time <= 0:
            continue
        idx = chunk.index
        out.loc[idx, "case_speedup"] = baseline_time / chunk["time_sec"]

    return out


def compute_determinism(det1: pd.DataFrame, det2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if det1.empty or det2.empty:
        return pd.DataFrame(), pd.DataFrame()

    left = det1.copy()
    right = det2.copy()

    if left.shape[1] < 4 or right.shape[1] < 4:
        return pd.DataFrame(), pd.DataFrame()

    left = left.iloc[:, :4]
    right = right.iloc[:, :4]
    left.columns = ["i", "j", "k", "value_run1"]
    right.columns = ["i", "j", "k", "value_run2"]

    merged = left.merge(right, on=["i", "j", "k"], how="outer")
    merged["value_run1"] = pd.to_numeric(merged["value_run1"], errors="coerce")
    merged["value_run2"] = pd.to_numeric(merged["value_run2"], errors="coerce")
    merged["delta_abs"] = (merged["value_run1"] - merged["value_run2"]).abs()

    if merged["delta_abs"].notna().sum() == 0:
        return merged, pd.DataFrame()

    stats = pd.DataFrame(
        [
            {
                "count": int(merged["delta_abs"].count()),
                "nonzero_count": int((merged["delta_abs"] > 0).sum()),
                "max_abs_delta": float(merged["delta_abs"].max()),
                "mean_abs_delta": float(merged["delta_abs"].mean()),
                "median_abs_delta": float(merged["delta_abs"].median()),
                "p95_abs_delta": float(merged["delta_abs"].quantile(0.95)),
                "p99_abs_delta": float(merged["delta_abs"].quantile(0.99)),
                "std_abs_delta": float(merged["delta_abs"].std(ddof=0)),
            }
        ]
    )

    return merged, stats


def save_plot(fig: Figure, stem: str, out_png: Path, out_pdf: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    png_path = out_png / f"{stem}.png"
    pdf_path = out_pdf / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    records.append({"asset_type": "figure", "name": stem, "format": "png", "path": str(png_path)})
    records.append({"asset_type": "figure", "name": stem, "format": "pdf", "path": str(pdf_path)})
    return records


def plot_speedup_efficiency_gflops(scal: pd.DataFrame, out_png: Path, out_pdf: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if scal.empty:
        return records

    for metric, ylabel, stem_prefix in [
        ("speedup_vs_serial", "Speedup vs Serial", "scalability_speedup"),
        ("efficiency", "Parallel Efficiency", "scalability_efficiency"),
        ("gflops", "GFLOP/s", "scalability_gflops"),
    ]:
        for shape in sorted(scal["shape_class"].dropna().unique()):
            fig, ax = plt.subplots(figsize=(8.8, 5.2))
            sub_shape = scal[(scal["shape_class"] == shape) & (scal["mode"] != "serial")]
            if sub_shape.empty:
                plt.close(fig)
                continue

            for mode in ["omp", "cuda", "hybrid"]:
                mode_df = sub_shape[sub_shape["mode"] == mode]
                if mode_df.empty:
                    continue
                agg = mode_df.groupby("threads", as_index=False)[metric].median().sort_values("threads")
                ax.plot(
                    agg["threads"],
                    agg[metric],
                    marker="o",
                    linewidth=2,
                    label=mode.upper(),
                    color=MODE_COLORS.get(mode, "#333333"),
                )

            ax.axvline(36, linestyle="--", color="#6b7280", linewidth=1)
            ax.axvspan(36, max(float(sub_shape["threads"].max()), 36), color="#fef3c7", alpha=0.35)
            ax.set_title(f"{ylabel} by Threads ({shape.capitalize()} tensors)")
            ax.set_xlabel("Threads")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            records.extend(save_plot(fig, f"{stem_prefix}_{shape}", out_png, out_pdf))

    return records


def plot_scalability_size_and_oversub(scal: pd.DataFrame, out_png: Path, out_pdf: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if scal.empty:
        return records

    # Time vs input storage size
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for mode in MODE_ORDER:
        mode_df = scal[scal["mode"] == mode]
        if mode_df.empty:
            continue
        preferred_threads = 36 if mode in {"omp", "hybrid"} else 1
        mode_df = mode_df[mode_df["threads"] == preferred_threads]
        if mode_df.empty:
            continue
        agg = mode_df.groupby("storage_size_input", as_index=False)["time_sec"].median().sort_values("storage_size_input")
        ax.plot(agg["storage_size_input"], agg["time_sec"], marker="o", label=f"{mode.upper()} (t={preferred_threads})", color=MODE_COLORS.get(mode, "#333333"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input storage size (elements)")
    ax.set_ylabel("Time (sec)")
    ax.set_title("Runtime Scaling vs Tensor Size")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    records.extend(save_plot(fig, "scalability_time_vs_storage", out_png, out_pdf))

    # Oversubscription trend on the largest shape available
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for shape in sorted(scal["shape_class"].dropna().unique()):
        for mode in ["omp", "hybrid"]:
            mode_df = scal[(scal["shape_class"] == shape) & (scal["mode"] == mode)]
            if mode_df.empty:
                continue
            largest_size = mode_df["storage_size_input"].max()
            trace = mode_df[mode_df["storage_size_input"] == largest_size].sort_values("threads")
            if trace.empty:
                continue
            ax.plot(
                trace["threads"],
                trace["time_sec"],
                marker="o",
                linewidth=1.8,
                label=f"{shape.capitalize()} {mode.upper()}",
            )

    ax.axvline(36, linestyle="--", color="#6b7280", linewidth=1)
    ax.axvspan(36, max(float(scal["threads"].max()), 36), color="#fee2e2", alpha=0.35)
    ax.annotate("OOM observed near t=64 for tall OMP", xy=(64, ax.get_ylim()[0]), xytext=(42, ax.get_ylim()[0] * 1.25 if ax.get_ylim()[0] > 0 else 1), fontsize=9)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Time (sec)")
    ax.set_title("Oversubscription Behavior on Largest Tensor per Shape")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    records.extend(save_plot(fig, "scalability_oversubscription_behavior", out_png, out_pdf))

    return records


def plot_load_imbalance(load_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if load_df.empty:
        return records

    load_df = load_df.copy()
    load_df["mode"] = load_df["mode"].map(normalize_mode)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    load_df["label"] = load_df["shape_class"].str.capitalize() + "-" + load_df["mode"].str.upper()
    load_df_sorted = load_df.sort_values("sweep_load_imbalance", ascending=False)
    ax.bar(load_df_sorted["label"], load_df_sorted["sweep_load_imbalance"], color="#6366f1")
    ax.set_ylabel("Sweep load imbalance")
    ax.set_title("Load Imbalance Across Thread Sweeps")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(True, axis="y", alpha=0.25)
    records.extend(save_plot(fig, "scalability_load_imbalance_sweep", out_png, out_pdf))

    return records


def plot_storage_compression(scal: pd.DataFrame, out_png: Path, out_pdf: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if scal.empty:
        return records

    # Compression distribution by shape
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    groups = []
    labels = []
    for shape in sorted(scal["shape_class"].dropna().unique()):
        vals = scal.loc[scal["shape_class"] == shape, "input_compression_dense_to_active"].dropna().values
        if len(vals) == 0:
            continue
        groups.append(vals)
        labels.append(shape.capitalize())
    if groups:
        positions = list(range(1, len(groups) + 1))
        ax.boxplot(groups, positions=positions, showfliers=False)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Compression ratio (dense to active)")
        ax.set_title("Storage Compression by Shape Class")
        ax.grid(True, axis="y", alpha=0.25)
        records.extend(save_plot(fig, "storage_compression_by_shape", out_png, out_pdf))
    else:
        plt.close(fig)

    # ELL utilization by shape + mode
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    agg = (
        scal.groupby(["shape_class", "mode"], as_index=False)["input_ell_slot_utilization"]
        .median()
        .sort_values(["shape_class", "mode"])
    )
    if not agg.empty:
        agg["label"] = agg["shape_class"].str.capitalize() + "-" + agg["mode"].str.upper()
        ax.bar(agg["label"], agg["input_ell_slot_utilization"], color="#10b981")
        ax.set_ylabel("ELL slot utilization")
        ax.set_title("Median ELL Slot Utilization")
        ax.tick_params(axis="x", rotation=70)
        ax.grid(True, axis="y", alpha=0.25)
        records.extend(save_plot(fig, "storage_ell_slot_utilization", out_png, out_pdf))
    else:
        plt.close(fig)

    return records


def plot_correctness(corr: pd.DataFrame, out_png: Path, out_pdf: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if corr.empty:
        return records

    for op in ["hadamard", "qr", "svd"]:
        sub = corr[corr["operation"] == op]
        if sub.empty:
            continue

        # Speedup grouped by density combo and mode
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        dens = sorted(sub["density_combo"].dropna().unique())
        x = np.arange(len(dens), dtype=float)
        width = 0.19

        for idx, mode in enumerate(MODE_ORDER):
            mode_vals = []
            for d in dens:
                value = sub[(sub["density_combo"] == d) & (sub["mode"] == mode)]["speedup_vs_serial"].median()
                mode_vals.append(value)
            ax.bar(x + (idx - 1.5) * width, mode_vals, width=width, label=mode.upper(), color=MODE_COLORS.get(mode, "#444444"))

        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("_", "-") for d in dens])
        ax.set_ylabel("Median speedup vs serial")
        ax.set_title(f"{op.upper()} Speedup by Tensor Type and Mode")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="best")
        records.extend(save_plot(fig, f"correctness_speedup_{op}", out_png, out_pdf))

        # Runtime grouped by mode
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        run_agg = sub.groupby("mode", as_index=False)["time_sec"].median()
        run_agg = run_agg.set_index("mode").reindex(MODE_ORDER).reset_index()
        ax.bar(run_agg["mode"].str.upper(), run_agg["time_sec"], color=[MODE_COLORS.get(m, "#444444") for m in run_agg["mode"]])
        ax.set_yscale("log")
        ax.set_ylabel("Median runtime (sec)")
        ax.set_title(f"{op.upper()} Median Runtime by Mode")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        records.extend(save_plot(fig, f"correctness_runtime_{op}", out_png, out_pdf))

    # COO vs ELL ratio (ELL / COO)
    sub = corr[corr["storage_format"].isin(["coo", "ell"])].copy()
    if not sub.empty:
        pivot = sub.pivot_table(
            index=["density_combo", "operation", "mode", "tensor_shape"],
            columns="storage_format",
            values="time_sec",
            aggfunc="median",
        ).reset_index()
        if "coo" in pivot.columns and "ell" in pivot.columns:
            pivot["ell_over_coo"] = pivot["ell"] / pivot["coo"]
            fig, ax = plt.subplots(figsize=(8.8, 5.2))
            vals = [pivot[pivot["operation"] == op]["ell_over_coo"].dropna().values for op in ["hadamard", "qr", "svd"]]
            labels = ["Hadamard", "QR", "SVD"]
            vals = [v for v in vals if len(v) > 0]
            if vals:
                positions = list(range(1, len(vals) + 1))
                ax.boxplot(vals, positions=positions, showfliers=False)
                ax.set_xticks(positions)
                ax.set_xticklabels(labels[: len(vals)])
                ax.axhline(1.0, linestyle="--", color="#6b7280", linewidth=1)
                ax.set_ylabel("Runtime ratio (ELL / COO)")
                ax.set_title("Storage Format Impact on Runtime")
                ax.grid(True, axis="y", alpha=0.25)
                records.extend(save_plot(fig, "correctness_storage_ratio_ell_over_coo", out_png, out_pdf))
            else:
                plt.close(fig)

    return records


def plot_determinism(det_joined: pd.DataFrame, out_png: Path, out_pdf: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if det_joined.empty or "delta_abs" not in det_joined:
        return records

    delta = det_joined["delta_abs"].dropna()
    if delta.empty:
        return records

    # Histogram (log-scaled x)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    eps = 1e-20
    delta_safe = np.clip(delta.values, eps, None)
    bins = np.logspace(np.log10(delta_safe.min()), np.log10(delta_safe.max() if delta_safe.max() > eps else eps * 10), 50)
    ax.hist(delta_safe, bins=bins, color="#7c3aed", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Absolute delta between run1 and run2")
    ax.set_ylabel("Count")
    ax.set_title("Determinism Delta Distribution")
    ax.grid(True, which="both", alpha=0.25)
    records.extend(save_plot(fig, "determinism_delta_histogram", out_png, out_pdf))

    # CDF
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    sorted_vals = np.sort(delta_safe)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, color="#7c3aed", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Absolute delta between run1 and run2")
    ax.set_ylabel("CDF")
    ax.set_title("Determinism Delta CDF")
    ax.grid(True, which="both", alpha=0.25)
    records.extend(save_plot(fig, "determinism_delta_cdf", out_png, out_pdf))

    return records


def plot_bottleneck(profile: pd.DataFrame, out_png: Path, out_pdf: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if profile.empty:
        return records

    # Sparse scaling for case1 and case3
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for case_name, color in [("case1_sparse_omp", "#2563eb"), ("case3_large_sparse_omp", "#dc2626")]:
        sub = profile[profile["case"] == case_name].copy()
        sub = sub.sort_values("threads_or_ratio_num")
        if sub.empty:
            continue
        base = sub["time_sec"].iloc[0]
        if base <= 0:
            continue
        speedup = base / sub["time_sec"]
        ax.plot(sub["threads_or_ratio_num"], speedup, marker="o", linewidth=2, color=color, label=case_name)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup vs case baseline")
    ax.set_title("Sparse OMP Bottleneck Scaling")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    records.extend(save_plot(fig, "profile_sparse_scaling_cases", out_png, out_pdf))

    # Hybrid ratio sensitivity case4
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    sub = profile[profile["case"] == "case4_hybrid_ratio"].copy()
    sub = sub.sort_values("threads_or_ratio_num")
    if not sub.empty:
        ax.plot(sub["threads_or_ratio_num"], sub["time_sec"], marker="o", linewidth=2, color="#059669")
        ax.set_xlabel("GPU ratio")
        ax.set_ylabel("Time (sec)")
        ax.set_title("Hybrid GPU Ratio Sensitivity (Case 4)")
        ax.grid(True, alpha=0.3)
        records.extend(save_plot(fig, "profile_case4_gpu_ratio_sensitivity", out_png, out_pdf))
    else:
        plt.close(fig)

    # Case-wise timing overview
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    agg = profile.groupby(["case", "mode"], as_index=False)["time_sec"].median().sort_values("time_sec", ascending=False)
    if not agg.empty:
        labels = agg["case"] + "-" + agg["mode"].str.upper()
        ax.bar(labels, agg["time_sec"], color="#f59e0b")
        ax.set_ylabel("Median time (sec)")
        ax.set_title("Profiling Case Timings")
        ax.tick_params(axis="x", rotation=70)
        ax.grid(True, axis="y", alpha=0.25)
        records.extend(save_plot(fig, "profile_case_timing_overview", out_png, out_pdf))
    else:
        plt.close(fig)

    # Memory vs time scatter
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    if "working_set_mem_mib" in profile and "time_sec" in profile:
        for case_name, chunk in profile.groupby("case"):
            ax.scatter(chunk["working_set_mem_mib"], chunk["time_sec"], label=case_name, alpha=0.8)
        ax.set_xlabel("Working set memory (MiB)")
        ax.set_ylabel("Time (sec)")
        ax.set_title("Profiling Memory vs Runtime")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        records.extend(save_plot(fig, "profile_memory_vs_runtime", out_png, out_pdf))
    else:
        plt.close(fig)

    return records


def plot_log_derived(logs_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if logs_df.empty:
        return records

    sub = logs_df.dropna(subset=["time_sec"]).copy()
    sub["mode"] = sub["mode"].map(normalize_mode)

    # Runtime distribution by mode
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    data = [sub[sub["mode"] == mode]["time_sec"].dropna().values for mode in MODE_ORDER]
    data = [d for d in data if len(d) > 0]
    labels = [m.upper() for m in MODE_ORDER if len(sub[sub["mode"] == m]) > 0]
    if data:
        positions = list(range(1, len(data) + 1))
        ax.boxplot(data, positions=positions, showfliers=False)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_yscale("log")
        ax.set_ylabel("Time (sec)")
        ax.set_title("Operation Log Runtime Distribution by Mode")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        records.extend(save_plot(fig, "logs_runtime_distribution_by_mode", out_png, out_pdf))
    else:
        plt.close(fig)

    # Dense vs sparse slice contribution scatter
    if "dense_slices" in sub.columns and "sparse_slices" in sub.columns:
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        ax.scatter(sub["dense_slices"], sub["sparse_slices"], alpha=0.7, color="#7c3aed")
        ax.set_xlabel("Dense slices")
        ax.set_ylabel("Sparse slices")
        ax.set_title("Dense/Sparse Slice Mix Across Logs")
        ax.grid(True, alpha=0.25)
        records.extend(save_plot(fig, "logs_dense_sparse_slice_mix", out_png, out_pdf))
    return records


def plot_slurm_size_catalog(size_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if size_df.empty:
        return records

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for shape, chunk in size_df.groupby("shape_class"):
        ax.plot(chunk["storage_size_input"], chunk["working_set_mem_mib"], marker="o", linewidth=2, label=f"{shape.capitalize()} working")
        ax.plot(chunk["storage_size_input"], chunk["dense_equiv_working_set_mem_mib"], marker="x", linestyle="--", linewidth=1.2, label=f"{shape.capitalize()} dense-eq")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input storage size")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title("Memory Scaling from Script [SIZE] Outputs")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    records.extend(save_plot(fig, "slurm_size_memory_scaling", out_png, out_pdf))

    return records


def export_tables(
    scal: pd.DataFrame,
    load_df: pd.DataFrame,
    corr: pd.DataFrame,
    profile: pd.DataFrame,
    det_joined: pd.DataFrame,
    det_stats: pd.DataFrame,
    logs_df: pd.DataFrame,
    size_df: pd.DataFrame,
    events_df: pd.DataFrame,
    out_detailed: Path,
    out_summary: Path,
) -> List[Dict[str, str]]:
    table_records: List[Dict[str, str]] = []

    def save_table(df: pd.DataFrame, name: str, folder: Path, latex: bool = False) -> None:
        path = folder / f"{name}.csv"
        df.to_csv(path, index=False)
        table_records.append({"asset_type": "table", "name": name, "format": "csv", "path": str(path)})
        if latex:
            latex_path = folder / f"{name}.tex"
            with latex_path.open("w", encoding="utf-8") as f:
                f.write(df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.6g}" if isinstance(x, (int, float)) else str(x)))
            table_records.append({"asset_type": "table", "name": name, "format": "tex", "path": str(latex_path)})

    # Detailed tables
    save_table(scal, "detailed_scalability_enriched", out_detailed)
    save_table(load_df, "detailed_load_imbalance", out_detailed)
    save_table(corr, "detailed_correctness_enriched", out_detailed)
    save_table(profile, "detailed_profile_bottleneck_enriched", out_detailed)
    save_table(det_joined, "detailed_determinism_joined", out_detailed)
    save_table(det_stats, "detailed_determinism_stats", out_detailed)
    save_table(logs_df, "detailed_log_metrics_all", out_detailed)
    save_table(size_df, "detailed_slurm_size_catalog", out_detailed)
    save_table(events_df, "detailed_slurm_events", out_detailed)

    # Summary tables
    if not scal.empty:
        best_speedup = (
            scal[scal["mode"] != "serial"]
            .sort_values("speedup_vs_serial", ascending=False)
            .groupby(["shape_class", "mode"], as_index=False)
            .first()[["shape_class", "mode", "threads", "tensor_shape", "speedup_vs_serial", "efficiency", "gflops"]]
        )
        save_table(best_speedup, "summary_best_speedup_by_shape_mode", out_summary, latex=True)

        compression_summary = (
            scal.groupby(["shape_class", "mode"], as_index=False)[
                [
                    "input_compression_dense_to_coo",
                    "input_compression_dense_to_ell",
                    "input_compression_dense_to_active",
                    "input_ell_slot_utilization",
                    "memory_ratio_working_vs_dense",
                ]
            ]
            .median()
            .sort_values(["shape_class", "mode"])
        )
        save_table(compression_summary, "summary_storage_compression", out_summary, latex=True)

    if not corr.empty:
        rank_corr = (
            corr.groupby(["density_combo", "operation", "mode"], as_index=False)["time_sec"].median()
            .sort_values(["density_combo", "operation", "time_sec"])
        )
        fastest = rank_corr.groupby(["density_combo", "operation"], as_index=False).first()
        save_table(fastest, "summary_fastest_mode_per_operation_density", out_summary, latex=True)

    if not profile.empty:
        profile_summary = (
            profile.groupby(["case", "mode"], as_index=False)["time_sec"].agg(["min", "median", "max"]).reset_index()
        )
        save_table(profile_summary, "summary_profile_case_timing_stats", out_summary, latex=True)

    if not load_df.empty:
        load_summary = load_df[[
            "mode",
            "shape_class",
            "tensor_shape",
            "min_time_sec",
            "max_time_sec",
            "sweep_load_imbalance",
            "best_threads",
            "worst_threads",
        ]].copy()
        save_table(load_summary, "summary_load_imbalance_overview", out_summary, latex=True)

    if not det_stats.empty:
        save_table(det_stats, "summary_determinism_stats", out_summary, latex=True)

    if not events_df.empty:
        anomaly_summary = events_df.groupby(["source_file", "severity", "marker"], as_index=False).size()
        save_table(anomaly_summary, "summary_anomalies", out_summary, latex=True)

    return table_records


def ensure_dirs(base_out: Path) -> Dict[str, Path]:
    dirs = {
        "root": base_out,
        "fig_png": base_out / "figures" / "png",
        "fig_pdf": base_out / "figures" / "pdf",
        "tables_detailed": base_out / "tables" / "detailed",
        "tables_summary": base_out / "tables" / "summary",
        "metadata": base_out / "metadata",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def build_assets(root: Path, out_root: Path, dpi: int = 220) -> None:
    global PLOT_DPI
    PLOT_DPI = dpi
    plt.style.use("seaborn-v0_8-whitegrid")

    dirs = ensure_dirs(out_root)

    # Core datasets
    scal_path = root / "artifacts" / "scalability" / "scalability_matrix.csv"
    load_path = root / "artifacts" / "scalability" / "load_imbalance.csv"
    corr_path = root / "artifacts" / "debug" / "correctness_results.csv"
    bottleneck_path = root / "artifacts" / "profile" / "bottleneck" / "bottleneck_metrics.csv"
    det1_path = root / "artifacts" / "debug" / "determinism_run1.csv"
    det2_path = root / "artifacts" / "debug" / "determinism_run2.csv"

    scal = enrich_scalability(read_csv_safe(scal_path))
    load_df = read_csv_safe(load_path)
    corr = enrich_correctness(read_csv_safe(corr_path))
    bottleneck = enrich_bottleneck(read_csv_safe(bottleneck_path))

    det1 = read_csv_safe(det1_path)
    det2 = read_csv_safe(det2_path)
    det_joined, det_stats = compute_determinism(det1, det2)

    # Parse all logs under artifacts
    log_files = sorted((root / "artifacts").rglob("*.log"))
    parsed_logs = [parse_operation_log(p, root) for p in log_files]
    logs_df = coerce_numeric_like(pd.DataFrame(parsed_logs))
    if "mode" in logs_df.columns:
        logs_df["mode"] = logs_df["mode"].map(normalize_mode)

    # Parse script outputs
    size_df = parse_slurm_size_catalog(root / "run_23295.out")
    events_df = parse_slurm_events(
        [
            root / "run_23295.out",
            root / "run_23295.err",
            root / "profile_23296.out",
            root / "profile_23296.err",
            root / "debug_23294.out",
            root / "debug_23294.err",
        ]
    )

    # Charts
    figure_records: List[Dict[str, str]] = []
    figure_records.extend(plot_speedup_efficiency_gflops(scal, dirs["fig_png"], dirs["fig_pdf"]))
    figure_records.extend(plot_scalability_size_and_oversub(scal, dirs["fig_png"], dirs["fig_pdf"]))
    figure_records.extend(plot_load_imbalance(load_df, dirs["fig_png"], dirs["fig_pdf"]))
    figure_records.extend(plot_storage_compression(scal, dirs["fig_png"], dirs["fig_pdf"]))
    figure_records.extend(plot_correctness(corr, dirs["fig_png"], dirs["fig_pdf"]))
    figure_records.extend(plot_determinism(det_joined, dirs["fig_png"], dirs["fig_pdf"]))
    figure_records.extend(plot_bottleneck(bottleneck, dirs["fig_png"], dirs["fig_pdf"]))
    figure_records.extend(plot_log_derived(logs_df, dirs["fig_png"], dirs["fig_pdf"]))
    figure_records.extend(plot_slurm_size_catalog(size_df, dirs["fig_png"], dirs["fig_pdf"]))

    # Tables
    table_records = export_tables(
        scal,
        load_df,
        corr,
        bottleneck,
        det_joined,
        det_stats,
        logs_df,
        size_df,
        events_df,
        dirs["tables_detailed"],
        dirs["tables_summary"],
    )

    # Manifests
    figure_manifest = pd.DataFrame(figure_records)
    table_manifest = pd.DataFrame(table_records)

    figure_manifest.to_csv(dirs["metadata"] / "figure_manifest.csv", index=False)
    table_manifest.to_csv(dirs["metadata"] / "table_manifest.csv", index=False)

    dataset_manifest = pd.DataFrame(
        [
            {"name": "scalability_matrix", "path": str(scal_path), "rows": int(len(scal))},
            {"name": "load_imbalance", "path": str(load_path), "rows": int(len(load_df))},
            {"name": "correctness_results", "path": str(corr_path), "rows": int(len(corr))},
            {"name": "bottleneck_metrics", "path": str(bottleneck_path), "rows": int(len(bottleneck))},
            {"name": "determinism_joined", "path": "merged(run1,run2)", "rows": int(len(det_joined))},
            {"name": "all_operation_logs", "path": "artifacts/**.log", "rows": int(len(logs_df))},
            {"name": "slurm_size_catalog", "path": str(root / "run_23295.out"), "rows": int(len(size_df))},
            {"name": "slurm_events", "path": "run/debug/profile out+err", "rows": int(len(events_df))},
        ]
    )
    dataset_manifest.to_csv(dirs["metadata"] / "dataset_manifest.csv", index=False)

    readme = dirs["root"] / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Report Assets",
                "",
                "This directory is generated by scripts/generate_report_assets.py.",
                "",
                "## Contents",
                "- figures/png: chart exports for report insertion",
                "- figures/pdf: vector figure exports",
                "- tables/detailed: full detailed tables",
                "- tables/summary: compact report tables and LaTeX tabular files",
                "- metadata: manifests and dataset index",
                "",
                "## Notes",
                "- Oversubscription region is highlighted beyond 36 threads.",
                "- OOM/Killed events are preserved in anomaly tables.",
                "- Nsight Compute permission warnings are captured in slurm event tables.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Generated figures: {len(figure_manifest)}")
    print(f"Generated table artifacts: {len(table_manifest)}")
    print(f"Output directory: {dirs['root']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all graphs, charts, and tables from experiment artifacts.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root path")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <root>/artifacts/report_assets)",
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = args.out.resolve() if args.out else (root / "artifacts" / "report_assets")
    build_assets(root=root, out_root=out, dpi=args.dpi)


if __name__ == "__main__":
    main()
