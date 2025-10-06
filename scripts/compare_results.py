import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# Folder where your CSVs live
DATA_DIR = "results/csv"

def load_csvs(pattern):
    """Load all CSVs that match a filename pattern (e.g., 'top_left_*fps.csv')."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not files:
        print(f"❌ No CSV files found matching {pattern} in {DATA_DIR}")
        return {}

    data = {}
    for f in files:
        try:
            fps = os.path.basename(f).split("_")[-1].replace("fps.csv", "")
            df = pd.read_csv(f)

            # Quick sanity check
            if "area" not in df.columns or "time_sec" not in df.columns:
                print(f"⚠️ Skipping {f} (missing expected columns)")
                continue

            print(f"✅ Loaded {f} ({len(df)} rows)")
            data[fps] = df
        except Exception as e:
            print(f"❌ Error loading {f}: {e}")

    return data

def plot_raw_area(data, out_png="results/csv/comparison_area_raw.png"):
    """Plot raw area over time for each fps setting."""
    if not data:
        print("⚠️ No data to plot (raw)")
        return

    plt.figure(figsize=(10,6))
    for fps, df in data.items():
        plt.plot(df["time_sec"], df["area"], label=f"{fps} fps", alpha=0.6)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Detected area")
    plt.title("Rat activity (area) comparison across FPS settings (raw)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"📊 Saved raw comparison plot: {out_png}")

def plot_avg_area(data, out_png="results/csv/comparison_area_avg.png"):
    """Plot average area per minute for each fps setting."""
    if not data:
        print("⚠️ No data to plot (avg)")
        return

    plt.figure(figsize=(10,6))
    for fps, df in data.items():
        df["minute"] = (df["time_sec"] // 60).astype(int)
        avg_area = df.groupby("minute")["area"].mean()
        plt.plot(avg_area.index, avg_area.values, label=f"{fps} fps")

    plt.xlabel("Time (minutes)")
    plt.ylabel("Average detected area")
    plt.title("Rat activity (area) comparison across FPS settings (avg per min)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"📊 Saved averaged comparison plot: {out_png}")

def main():
    print("🔍 Loading CSVs for top_left quadrant...")
    data = load_csvs("top_left_*fps.csv")

    # Plot raw
    plot_raw_area(data)

    # Plot averaged per minute
    plot_avg_area(data)

if __name__ == "__main__":
    main()
