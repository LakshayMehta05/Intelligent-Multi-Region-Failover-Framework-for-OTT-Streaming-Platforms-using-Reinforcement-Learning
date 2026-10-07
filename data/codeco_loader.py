"""
Loads the CODECO Media Consumption Traffic dataset (real HTTP streaming
traces from the EU Horizon CODECO project, captured during actual video
streaming sessions across a Greece origin CDN and Madrid edge cache).

Source: https://zenodo.org/records/19006500
Real fields used:
    time_ms        -> total request duration, used for latency
    wait_ms         -> server processing time (TTFB), supplementary
    throughput_Bps  -> real throughput, inverse-normalized into stall risk
    is_error        -> real HTTP error flag, used directly as error signal
"""

import pandas as pd
import numpy as np


def load_codeco_regions(csv_path: str = "data/codeco_media.csv") -> list:
    """Load the CODECO CSV and split it into one real trace per unique server node."""
    df = pd.read_csv(csv_path)
    hostnames = sorted(df["hostname"].dropna().unique())

    traces = []
    for host in hostnames:
        sub = df[df["hostname"] == host].reset_index(drop=True)
        traces.append(_build_trace(sub, host))
    return traces, hostnames


def _build_trace(df: pd.DataFrame, host_name: str) -> dict:
    time_ms = pd.to_numeric(df["time_ms"], errors="coerce").fillna(df["time_ms"].median())
    throughput = pd.to_numeric(df["throughput_Bps"], errors="coerce").fillna(0.0)
    is_error = pd.to_numeric(df["is_error"], errors="coerce").fillna(0.0)

    # Latency: clip extreme outliers (max was 217s, a clear anomaly) using 95th percentile cap
    p95 = time_ms.quantile(0.95)
    time_ms_capped = time_ms.clip(upper=max(p95, 100))
    max_val = time_ms_capped.max()
    if max_val < 1e-6:
        latency_norm = np.full(len(df), 0.2)
    else:
        latency_norm = (time_ms_capped / max_val).clip(0.0, 1.0).fillna(0.2).to_numpy()

    # Error rate: real HTTP error flag, smoothed slightly so it's not purely binary per-row
    error_norm = is_error.rolling(window=5, min_periods=1).mean().clip(0.0, 1.0).to_numpy()

    # Stall risk: inverse of throughput, percentile-normalized per node
    p10, p90 = throughput.quantile(0.1), throughput.quantile(0.9)
    if p90 - p10 < 1e-6:
        stall_norm = np.full(len(df), 0.3)
    else:
        stall_norm = (1.0 - (throughput - p10) / (p90 - p10)).clip(0.0, 1.0).to_numpy()

    return {
        "latency": latency_norm,
        "error_rate": error_norm,
        "buffer_stall": stall_norm,
        "length": len(df),
        "host": host_name,
    }


if __name__ == "__main__":
    traces, hostnames = load_codeco_regions()
    print(f"Loaded {len(traces)} region traces from hosts: {hostnames}")
    for t in traces:
        print(f"\n{t['host']} ({t['length']} rows):")
        print(f"  latency mean={t['latency'].mean():.2f}, error mean={t['error_rate'].mean():.2f}, stall mean={t['buffer_stall'].mean():.2f}")