import pandas as pd
import argparse
import os


def merge_with_labels(x_csv, y_csv, output_csv):
    X = pd.read_csv(x_csv)
    y = pd.read_csv(y_csv)

    n = min(len(X), len(y))
    if len(X) != len(y):
        print(f"Row count mismatch - X: {len(X)}, y: {len(y)}. Using first {n} rows.")

    merged = X.iloc[:n].copy().reset_index(drop=True)
    merged["stage"] = y["stage"].iloc[:n].values

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print(f"Rows in output : {len(merged)}")
    print(f"Output saved   : {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge binned rat-tracking CSV with y-train labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("x_csv",      help="Path to the binned X data CSV")
    parser.add_argument("y_csv",      help="Path to the y-train labels CSV (single 'stage' column)")
    parser.add_argument("output_csv", help="Path for the merged output CSV")
    args = parser.parse_args()

    merge_with_labels(args.x_csv, args.y_csv, args.output_csv)


if __name__ == "__main__":
    main()
