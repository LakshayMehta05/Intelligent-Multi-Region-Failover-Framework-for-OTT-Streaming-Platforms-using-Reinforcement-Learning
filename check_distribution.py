import sys
sys.path.insert(0, ".")
from data.trace_loader import load_region_traces
import numpy as np

trace_paths = [
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/Season3-StrangerThings/B_2019.11.26_13.50.48.csv",
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/Season3-StrangerThings/B_2019.12.03_08.02.05.csv",
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/animated-RickandMorty/B_2019.11.26_08.02.38.csv",
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/animated-RickandMorty/B_2019.11.28_08.02.19.csv",
]
traces = load_region_traces(trace_paths)
for i, t in enumerate(traces):
    stall = t["buffer_stall"]
    err = t["error_rate"]
    print(f"Region {i}:")
    print(f"  buffer_stall mean={stall.mean():.2f}, median={np.median(stall):.2f}, pct_above_0.8={np.mean(stall > 0.8) * 100:.1f}%")
    print(f"  error_rate   mean={err.mean():.2f}, pct_above_0.3={np.mean(err > 0.3) * 100:.1f}%")
    import pandas as pd
print("\n--- Raw DL_bitrate stats ---")
for p in trace_paths:
    df = pd.read_csv(p, skiprows=range(1, 201), nrows=2000)
    dl = pd.to_numeric(df["DL_bitrate"], errors="coerce").fillna(0.0)
    print(f"{p.split('/')[-1]}: mean={dl.mean():.0f}, median={dl.median():.0f}, p90={dl.quantile(0.9):.0f}, max={dl.max():.0f}")