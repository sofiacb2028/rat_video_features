# Is the rat sleeping or not

This repository contains scripts and example data for processing long rat videos and extracting quantitative features, inspired by the noninvasive sleep assessment paper (PMC8842275).

## Contents

- `extract_features.py` – Extracts frame-level shape features from video using OpenCV.
- `analyze_features.py` – Loads the CSV output and generates plots showing rat activity over time and distribution of movement sizes.
- `features.csv` – Example output CSV containing one row per frame with the following columns:
  - `frame` – Frame number
  - `time_sec` – Time in seconds
  - `area`, `perimeter`, `ellipse_major`, `ellipse_minor`, `ellipse_angle` – Contour/shape features
  - `hu1` … `hu7` – Hu moments (shape descriptors)
- `rat_activity_over_time.png` – Plot showing rat activity over time (contour area)
- `rat_area_histogram.png` – Histogram of rat contour area
