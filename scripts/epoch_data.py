import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import welch
import argparse
import os


EXCLUDED_COLS = ("epoch_id", "frame_num", "time_sec", "time_min")


def compute_psd_features(signal, fs):
    n = len(signal)

    if n < 4:
        print("Signal too small")

        return {k: np.nan for k in (
            "k_psd", "s_psd",
            "MPL_1", "MPL_3", "MPL_5",
            "Tot_PSD", "Max_PSD", "Min_PSD", "Ave_PSD", "Std_PSD", "MED_PSD",
            "TOP_SIGNAL",
        )}

    freqs, psd = welch(signal, fs=fs, nperseg=min(n, 64))

    def band_mean(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return psd[mask].mean() if mask.any() else np.nan

    return {

        "k_psd":      stats.kurtosis(psd),
        "s_psd":      stats.skew(psd),

        "MPL_1":      band_mean(0.1, 1.0),
        "MPL_3":      band_mean(1.0, 3.0),
        "MPL_5":      band_mean(3.0, 5.0),

        "Tot_PSD":    psd.sum(),
        "Max_PSD":    psd.max(),
        "Min_PSD":    psd.min(),
        "Ave_PSD":    psd.mean(),
        "Std_PSD":    psd.std(),
        "MED_PSD":    np.median(psd),

        "TOP_SIGNAL": freqs[np.argmax(psd)],
    }


def compute_time_features(signal):
    return {
        "Ave_Signal": signal.mean(),
        "Std_Signal": signal.std(),
        "Max_Signal": signal.max(),
        "Min_Signal": signal.min(),
        "MED_Signal": np.median(signal),
        "k":          stats.kurtosis(signal),
    }


def compute_epoch_features(epoch_df, numeric_cols, fs):
    row = {"epoch_id": epoch_df["epoch_id"].iloc[0]}

    for col in numeric_cols:
        signal = epoch_df[col].dropna().to_numpy(dtype=float)

        if len(signal) == 0:
            continue

        time_feats = compute_time_features(signal)
        psd_feats  = compute_psd_features(signal, fs)

        for feat_name, val in {**time_feats, **psd_feats}.items():
            row[f"{col}_{feat_name}"] = val

    return row


def bin_to_epochs(input_csv, output_csv, bin_size_sec=10, target_fps=15):
    df = pd.read_csv(input_csv)

    if df.empty:
        print("Error: Input CSV is empty.")
        return False

    print(f"Loaded {len(df):,} rows")
    print(f"Time range: {df['time_sec'].min():.1f}s – {df['time_sec'].max():.1f}s")
    print(f"Columns: {list(df.columns)}")

    df["epoch_id"] = (df["time_sec"] // bin_size_sec).astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in EXCLUDED_COLS]

    print(f"\nComputing features for columns: {numeric_cols}")
    print(f"Sampling rate: {target_fps} Hz  →  max resolvable frequency: {target_fps / 2} Hz")

    rows = []
    for epoch_id, epoch_df in df.groupby("epoch_id"):
        rows.append(compute_epoch_features(epoch_df, numeric_cols, fs=target_fps))

    result = pd.DataFrame(rows)

    result.insert(1, "time_sec", result["epoch_id"] * bin_size_sec)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    result.to_csv(output_csv, index=False)

    print(f"\nEpochs:          {len(result)}")
    print(f"Features/column: {len(result.columns) - 2}")
    print(f"Total columns:   {len(result.columns)}")
    print(f"Output saved:    {output_csv}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Bin per-frame rat-tracking CSV into N-second epochs with time & frequency features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_csv",  help="Path to the per-frame CSV from the detection script")
    parser.add_argument("output_csv", help="Path for the binned output CSV")
    parser.add_argument(
        "--bin-sec", type=float, default=10,
        help="Bin size in seconds (default: 10)"
    )
    parser.add_argument(
        "--fps", type=int, default=15,
        help="Frame rate of the source video, used as the PSD sampling frequency (default: 15)"
    )
    args = parser.parse_args()

    bin_to_epochs(args.input_csv, args.output_csv, bin_size_sec=args.bin_sec, target_fps=args.fps)


if __name__ == "__main__":
    main()
