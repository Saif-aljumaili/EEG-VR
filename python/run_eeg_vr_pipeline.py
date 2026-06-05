#!/usr/bin/env python3
"""
EEG-VR Dataset Processing Pipeline

This script reads DSI-7 EEG CSV recordings, performs quality-control checks,
filters the EEG (0.5-40 Hz), segments recordings into 1-second epochs, computes
FFT-based theta-band summaries (4-8 Hz), and generates representative figures.

Example:
    python python/run_eeg_vr_pipeline.py --raw-root "D:/EEG_VR_Prof.Adil/Saif/RAW EEGS" --out-root results/full_run --fs 300
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

CHANNELS = ["PzLE", "F4LE", "C4LE", "P4LE", "P3LE", "C3LE", "F3LE"]


def infer_sampling_rate(df: pd.DataFrame, fallback: float = 300.0) -> Tuple[float, str]:
    """Estimate sampling rate from timestamp columns; otherwise return fallback."""
    if "DeviceTimeStamp" in df.columns:
        t = pd.to_numeric(df["DeviceTimeStamp"], errors="coerce").dropna().to_numpy()
        if len(t) > 10:
            dt = np.diff(t[: min(len(t), 5000)])
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if len(dt):
                fs = 1.0 / np.median(dt)
                if 50 <= fs <= 2000:
                    return float(round(fs, 3)), "DeviceTimeStamp"

    if "DeviceTimeUnixTimeStamp" in df.columns:
        t = pd.to_numeric(df["DeviceTimeUnixTimeStamp"], errors="coerce").dropna().to_numpy()
        if len(t) > 10:
            dt = np.diff(t[: min(len(t), 5000)]) / 1000.0
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if len(dt):
                fs = 1.0 / np.median(dt)
                if 50 <= fs <= 2000:
                    return float(round(fs, 3)), "DeviceTimeUnixTimeStamp"

    return float(fallback), "fallback"


def discover_csv_files(raw_root: Path) -> pd.DataFrame:
    """Discover CSV files and infer group/condition from folder names."""
    rows = []
    for path in raw_root.rglob("*.csv"):
        parts = [p.lower() for p in path.parts]
        group = "unknown"
        condition = "unknown"
        if any("control" in p for p in parts):
            group = "control"
        if any("triathlete" in p or "athlete" in p for p in parts):
            group = "triathlete"
        if any(p == "baseline" or "baseline" in p for p in parts):
            condition = "baseline"
        if any(p == "vr" or p.endswith("/vr") or "vr" == p for p in parts):
            condition = "vr"
        rows.append({"filename": path.name, "path": str(path), "group": group, "condition": condition})
    return pd.DataFrame(rows).sort_values(["group", "condition", "filename"])


def read_eeg_csv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Read a CSV file and return full dataframe, EEG channel dataframe, and present channel names."""
    df = pd.read_csv(path)
    present = [ch for ch in CHANNELS if ch in df.columns]
    if not present:
        raise ValueError(f"No expected EEG channels were found in {path}")
    eeg = df[present].apply(pd.to_numeric, errors="coerce")
    return df, eeg, present


def bandpass_filter(x: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def compute_theta_summary(xf: np.ndarray, fs: float, channel_names: List[str]) -> pd.DataFrame:
    """Compute 1-second FFT-based theta-band absolute and relative power summaries."""
    nper = int(round(fs))
    n_epochs = xf.shape[0] // nper
    if n_epochs < 1:
        return pd.DataFrame()

    xtrim = xf[: n_epochs * nper, :]
    ep = xtrim.reshape(n_epochs, nper, xtrim.shape[1])
    ep = ep - ep.mean(axis=1, keepdims=True)

    freqs = np.fft.rfftfreq(nper, d=1 / fs)
    spec = np.abs(np.fft.rfft(ep, axis=1)) ** 2 / nper
    theta_idx = (freqs >= 4) & (freqs <= 8)
    total_idx = (freqs >= 0.5) & (freqs <= 40)

    theta_power = spec[:, theta_idx, :].sum(axis=1)
    total_power = spec[:, total_idx, :].sum(axis=1)
    rel_theta = theta_power / np.maximum(total_power, np.finfo(float).eps)

    rows = []
    for ci, ch in enumerate(channel_names):
        rows.append(
            {
                "channel": ch,
                "n_epochs_1s": int(n_epochs),
                "theta_abs_mean": float(theta_power[:, ci].mean()),
                "theta_abs_sd": float(theta_power[:, ci].std(ddof=1)) if n_epochs > 1 else 0.0,
                "theta_relative_mean": float(rel_theta[:, ci].mean()),
                "theta_relative_sd": float(rel_theta[:, ci].std(ddof=1)) if n_epochs > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_representative_figures(xf: np.ndarray, fs: float, channel_names: List[str], out_fig: Path) -> None:
    """Create a representative EEG segment and PSD figure from one recording."""
    out_fig.mkdir(parents=True, exist_ok=True)
    sec = 5
    ns = min(int(fs * sec), xf.shape[0])
    t = np.arange(ns) / fs

    plt.figure(figsize=(11, 6), dpi=200)
    offset = np.nanstd(xf[:ns, :]) * 6
    if not np.isfinite(offset) or offset == 0:
        offset = 100
    for ci, ch in enumerate(channel_names):
        y = xf[:ns, ci] + (len(channel_names) - ci - 1) * offset
        plt.plot(t, y, lw=0.8, color="black")
        plt.text(-0.08, (len(channel_names) - ci - 1) * offset, ch.replace("LE", ""), va="center", ha="right", fontsize=10)
    plt.xlabel("Time (s)")
    plt.yticks([])
    plt.title("Representative filtered EEG segment (0.5–40 Hz)")
    plt.xlim(0, sec)
    plt.tight_layout()
    plt.savefig(out_fig / "representative_eeg_segment.png", bbox_inches="tight")
    plt.close()

    f, pxx = signal.welch(xf, fs=fs, nperseg=int(fs * 2), axis=0)
    mean_psd = pxx.mean(axis=1)
    plt.figure(figsize=(9, 5.5), dpi=200)
    plt.semilogy(f, mean_psd, lw=1.8)
    plt.axvspan(4, 8, alpha=0.25)
    plt.text(4.3, np.nanmax(mean_psd[(f >= 4) & (f <= 8)]) * 1.2, "Theta band\n(4–8 Hz)", fontsize=10)
    plt.xlim(0, 40)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (a.u./Hz)")
    plt.title("Representative power spectral density")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_fig / "representative_psd_theta_band.png", bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EEG-VR quality-control and theta-band processing.")
    parser.add_argument("--raw-root", required=True, help="Path to RAW EEGS folder")
    parser.add_argument("--out-root", default="results/full_run", help="Output directory")
    parser.add_argument("--fs", type=float, default=300.0, help="Fallback sampling rate when timestamps are unavailable")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_tables = out_root / "tables"
    out_fig = out_root / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    files = discover_csv_files(raw_root)
    if files.empty:
        raise FileNotFoundError(f"No CSV files found under {raw_root}")
    files.to_csv(out_tables / "file_manifest.csv", index=False)

    qc_rows = []
    stats_rows = []
    theta_rows = []
    figure_saved = False

    for _, row in files.iterrows():
        path = Path(row["path"])
        df, eeg, present = read_eeg_csv(path)
        fs, fs_source = infer_sampling_rate(df, fallback=args.fs)
        n = len(df)
        duration = n / fs

        triggers = "not_available"
        if "Trigger" in df.columns:
            unique_triggers = pd.Series(df["Trigger"]).dropna().unique().tolist()
            triggers = ";".join(map(str, sorted(unique_triggers)[:20]))

        start_time = ""
        end_time = ""
        if "SystemTime" in df.columns and df["SystemTime"].dropna().shape[0] > 0:
            start_time = str(df["SystemTime"].dropna().iloc[0])
            end_time = str(df["SystemTime"].dropna().iloc[-1])

        qc_rows.append(
            {
                "filename": path.name,
                "group": row["group"],
                "condition": row["condition"],
                "n_samples": n,
                "n_channels_present": len(present),
                "channels_present": ";".join(present),
                "estimated_fs_hz": fs,
                "fs_source": fs_source,
                "duration_seconds": round(duration, 3),
                "duration_minutes": round(duration / 60, 3),
                "missing_eeg_values": int(eeg.isna().sum().sum()),
                "trigger_values": triggers,
                "start_time": start_time,
                "end_time": end_time,
            }
        )

        for ch in present:
            s = eeg[ch]
            stats_rows.append(
                {
                    "filename": path.name,
                    "group": row["group"],
                    "condition": row["condition"],
                    "channel": ch,
                    "mean_raw": float(s.mean()),
                    "sd_raw": float(s.std()),
                    "min_raw": float(s.min()),
                    "max_raw": float(s.max()),
                }
            )

        data = eeg.to_numpy(dtype=float)
        if np.isnan(data).any():
            med = np.nanmedian(data, axis=0)
            inds = np.where(np.isnan(data))
            data[inds] = np.take(med, inds[1])
        x = data - data.mean(axis=0, keepdims=True)
        xf = bandpass_filter(x, fs)

        theta = compute_theta_summary(xf, fs, present)
        if not theta.empty:
            theta.insert(0, "condition", row["condition"])
            theta.insert(0, "group", row["group"])
            theta.insert(0, "filename", path.name)
            theta_rows.append(theta)

        if not figure_saved:
            save_representative_figures(xf, fs, present, out_fig)
            figure_saved = True

    pd.DataFrame(qc_rows).to_csv(out_tables / "quality_control_summary.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(out_tables / "channel_raw_statistics.csv", index=False)
    if theta_rows:
        pd.concat(theta_rows, ignore_index=True).to_csv(out_tables / "theta_power_summary.csv", index=False)

    print(f"Processed {len(files)} CSV files.")
    print(f"Tables saved to: {out_tables}")
    print(f"Figures saved to: {out_fig}")


if __name__ == "__main__":
    main()
