import pandas as pd
import numpy as np
import argparse
import os


EXCLUDED_COLS = ("epoch_id", "frame_num", "time_sec", "time_min")


def bin_to_10_seconds(input_csv, output_csv, bin_size_sec=10):
    df = pd.read_csv(input_csv)

    if df.empty:
        print("Error: Input CSV is empty.")
        return False

    print(f"Loaded {len(df):,} rows")
    print(f"Time range: {df['time_sec'].min():.1f}s – {df['time_sec'].max():.1f}s")
    print(f"Columns: {list(df.columns)}")

    df["epoch_id"] = (df["time_sec"] // bin_size_sec).astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = list(filter(lambda col: col not in EXCLUDED_COLS, numeric_cols))

    epoch_groups = df.groupby("epoch_id")[numeric_cols].mean().reset_index()

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    epoch_groups.to_csv(output_csv, index=False)

    print(f"Epochs: {len(epoch_groups)}")
    print(f"Output saved: {output_csv}")
    print(f"Output columns: {list(epoch_groups.columns)}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Bin per-frame rat-tracking CSV into N-second averaged rows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_csv",  help="Path to the per-frame CSV from the detection script")
    parser.add_argument("output_csv", help="Path for the binned output CSV")
    parser.add_argument(
        "--bin-sec", type=float, default=10,
        help="Bin size in seconds (default: 10)"
    )
    args = parser.parse_args()

    bin_to_10_seconds(args.input_csv, args.output_csv, bin_size_sec=args.bin_sec)


if __name__ == "__main__":
    main()
