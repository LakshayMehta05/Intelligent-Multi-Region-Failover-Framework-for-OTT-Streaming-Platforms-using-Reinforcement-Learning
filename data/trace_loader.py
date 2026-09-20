"""
Loads real 5G network trace CSVs (from the UCC 5G production dataset,
recorded during real Netflix/Amazon Prime streaming sessions) and
converts them into normalized (0-1) health metrics compatible with
StreamFailoverEnv: latency, error_rate, buffer_stall_rate.

Source: https://github.com/uccmisl/5Gdataset
Real KPIs used per row:
    DL_bitrate  (kbps) -> inverse-normalized into a "stall risk" signal
    PINGAVG     (ms)   -> normalized into latency
    PINGLOSS    (%)    -> used directly as error_rate proxy

Notes on real-data quirks handled here:
1. PINGAVG/PINGLOSS are often unrecorded (all '-') in many sessions of
   this dataset. Where ping data is missing, we derive latency/error
   signals from DL_bitrate dynamics instead.
2. Raw per-second DL_bitrate is frequently 0 even during smooth playback,
   because streaming apps download in bursts then idle while the local
   buffer plays out. We apply a rolling average (30s window) to recover
   actual sustained throughput before normalizing.
3. Normalization uses each trace's own 10th/90th percentile range, so
   heterogeneous real sessions (which vary widely in absolute throughput)
   are all mapped onto a comparable, robust 0-1 scale.
"""

import pandas as pd
import numpy as np
import os


def load_trace(csv_path: str, max_rows: int = 2000, skip_start: int = 200, smooth_window: int = 30) -> dict:
    """Load one real trace CSV and return normalized metric arrays."""
    df = pd.read_csv(csv_path, skiprows=range(1, skip_start + 1), nrows=max_rows)

    dl_bitrate_raw = pd.to_numeric(df["DL_bitrate"], errors="coerce").fillna(0.0)
    dl_bitrate = dl_bitrate_raw.rolling(window=smooth_window, min_periods=1).mean()

    ping_avg = pd.to_numeric(df["PINGAVG"].replace("-", np.nan), errors="coerce")
    ping_loss = pd.to_numeric(df["PINGLOSS"].replace("-", np.nan), errors="coerce")

    p10, p90 = dl_bitrate.quantile(0.1), dl_bitrate.quantile(0.9)
    if p90 - p10 < 1e-6:
        stall_norm = np.full(len(df), 0.3)
    else:
        stall_norm = (1.0 - (dl_bitrate - p10) / (p90 - p10)).clip(0.0, 1.0).to_numpy()

    if ping_avg.notna().sum() > len(df) * 0.5:
        latency_norm = (ping_avg / 300.0).clip(0.0, 1.0).fillna(0.2).to_numpy()
        error_norm = (ping_loss / 100.0).clip(0.0, 1.0).fillna(0.02).to_numpy()
    else:
        latency_norm = np.clip(0.1 + 0.3 * stall_norm, 0.0, 1.0)
        error_norm = np.clip(0.02 + 0.15 * stall_norm, 0.0, 1.0)

    return {
        "latency": latency_norm,
        "error_rate": error_norm,
        "buffer_stall": stall_norm,
        "length": len(df),
    }


def load_region_traces(trace_paths: list) -> list:
    """Load one real trace per region. len(trace_paths) should equal n_regions."""
    return [load_trace(p) for p in trace_paths]


if __name__ == "__main__":
    base = "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/Season3-StrangerThings"
    files = [f for f in os.listdir(base) if f.endswith(".csv")]
    print(f"Found {len(files)} trace files in {base}")
    sample = load_trace(os.path.join(base, files[0]))
    print(f"Loaded {sample['length']} rows")
    print("Latency (first 5):", sample["latency"][:5])
    print("Error rate (first 5):", sample["error_rate"][:5])
    print("Buffer stall (first 5):", sample["buffer_stall"][:5])