import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV
df = pd.read_csv("features.csv")

# 2. Print the first few rows
print(df.head())
# number of rows and columns
print(df.shape)     


# 3. Plot activity over time (using 'area' as a proxy for movement)
plt.plot(df["time_sec"], df["area"])
plt.xlabel("Time (s)")
plt.ylabel("Contour Area")
plt.title("Rat Activity Over 4 Hours")
plt.savefig("rat_activity_over_time.png")  # saves as a PNG
plt.close()

# 4. Plot a histogram of movement sizes
df["area"].hist(bins=50)
plt.xlabel("Contour Area")
plt.ylabel("Count")
plt.title("Distribution of Rat Movement Sizes")
plt.savefig("rat_area_histogram.png")  # saves as a PNG
plt.close()

print("Analysis done! Two figures saved: 'rat_activity_over_time.png' and 'rat_area_histogram.png'")
