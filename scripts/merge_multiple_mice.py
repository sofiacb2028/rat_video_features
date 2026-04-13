import argparse
import glob
import sys
from pathlib import Path
import pandas as pd


def find_csvs(input_path: str | None) -> list[Path]:
    if input_path is None:
        files = sorted(Path(".").glob("*.csv"))
    else:
        p = Path(input_path)
        if p.is_dir():
            files = sorted(p.glob("*.csv"))
        elif p.is_file():
            files = [p]
        else:
            files = sorted(Path(f) for f in glob.glob(input_path))

    return [f for f in files if f.is_file()]


def merge(files: list[Path], output: Path, encoding: str = "utf-8") -> None:
    if not files:
        print("No CSV files found. Nothing to merge.")
        sys.exit(1)

    print(f"Found {len(files)} file(s) to merge:")
    for f in files:
        print(f"  {f}")

    dfs = []
    reference_columns = None

    for f in files:
        try:
            df = pd.read_csv(f, encoding=encoding)
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="latin-1")

        if reference_columns is None:
            reference_columns = list(df.columns)
        elif list(df.columns) != reference_columns:
            print(f"\nWARNING: '{f}' has different headers and will be skipped.")
            print(f"  Expected: {reference_columns}")
            print(f"  Got:      {list(df.columns)}")
            continue

        dfs.append(df)
        print(f"  Loaded {len(df):,} rows from {f.name}")

    if not dfs:
        print("No compatible files to merge.")
        sys.exit(1)

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(output, index=False, encoding="utf-8")

    print(f"\nDone! Merged {len(dfs)} file(s) → {len(merged):,} total rows")
    print(f"Output saved to: {output}")


def main():
    parser = argparse.ArgumentParser(description="Merge CSV files with identical headers.")
    parser.add_argument(
        "-i", "--input", nargs="+", metavar="PATH",
        help="Input CSV file(s) or a folder containing CSVs (default: current directory)"
    )
    parser.add_argument(
        "-o", "--output", default="merged.csv",
        help="Output filename (default: merged.csv)"
    )
    parser.add_argument(
        "--encoding", default="utf-8",
        help="File encoding (default: utf-8; falls back to latin-1 on error)"
    )
    args = parser.parse_args()

    if args.input is None:
        files = find_csvs(None)
    elif len(args.input) == 1:
        files = find_csvs(args.input[0])
    else:
        files = [Path(f) for f in args.input if Path(f).is_file()]

    output = Path(args.output)

    files = [f for f in files if f.resolve() != output.resolve()]

    merge(files, output, encoding=args.encoding)


if __name__ == "__main__":
    main()
