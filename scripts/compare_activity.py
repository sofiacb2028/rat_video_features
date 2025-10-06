import pandas as pd
import matplotlib.pyplot as plt
import os

# ====== CONFIG ======
rat_name = "top_left"  # change if needed
fps_list = [1, 2, 5]  # fps to compare
input_dir = "results"  # folder where your CSVs are
output_dir = "results"  # folder to save graphs
# ====================

dfs = {}
for fps in fps_list:
    csv_path = os.path.join(input_dir, f"{rat_name}_{fps}fps.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # compute simple activity = frame-to-frame difference in area
        df['activity'] = df['area'].diff().abs()
        dfs[fps] = df
    else:
        print(f"⚠️ CSV not found: {csv_path}")

# ---- Line plot of activity over time ----
plt.figure(figsize=(12,6))
for fps, df in dfs.items():
    plt.plot(df['time_sec'], df['activity'], label=f'{fps} fps')
plt.xlabel("Time (s)")
plt.ylabel("Activity (Δarea)")
plt.title(f"{rat_name} activity over time at different fps")
plt.legend()
plt.tight_layout()
line_plot_path = os.path.join(output_dir, f"{rat_name}_activity_comparison.png")
plt.savefig(line_plot_path)
plt.show()
print(f"✅ Line plot saved to {line_plot_path}")

# ---- Histogram of activity ----
plt.figure(figsize=(8,5))
for fps, df in dfs.items():
    plt.hist(df['activity'].dropna(), bins=50, alpha=0.5, label=f'{fps} fps')
plt.xlabel("Activity (Δarea)")
plt.ylabel("Frames")
plt.title(f"{rat_name} activity distribution at different fps")
plt.legend()
plt.tight_layout()
hist_path = os.path.join(output_dir, f"{rat_name}_activity_histogram.png")
plt.savefig(hist_path)
plt.show()
print(f"✅ Histogram saved to {hist_path}")
