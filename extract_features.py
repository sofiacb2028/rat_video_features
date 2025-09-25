import cv2
import numpy as np
import os
import csv

# --- Config ---
video_path = "top_left.mp4"  # Change this to your desired quadrant video
output_csv = "features.csv"
threshold_value = 60  # Adjust if needed
min_contour_area = 200  # Filter out small noise

# Morphological kernels
def diamond_kernel(size):
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    for i in range(size):
        for j in range(size):
            if abs(i - center) + abs(j - center) <= center:
                kernel[i, j] = 1
    return kernel

dilate_kernel = diamond_kernel(4)
erode_kernel = diamond_kernel(5)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("🚫 Could not open video!")
    exit()
else:
    print("✅ Video opened!")

fps = cap.get(cv2.CAP_PROP_FPS)

# Prepare CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = [
        "frame", "time_sec", "area", "perimeter",
        "ellipse_major", "ellipse_minor", "ellipse_angle",
        "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"
    ]
    writer.writerow(header)

    frame_num = 0
    while True:
        for _ in range(30):
            ret, frame = cap.read()
            print(f"Reading frame {frame_num}...", end="\r")

            if not ret or frame is None:
                print("⚠️ Failed to read frame!")
                cap.release()
                exit()


        frame_num += 30

        time_sec = frame_num / fps

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold to get mask
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Morphological filtering
        binary = cv2.dilate(binary, dilate_kernel)
        binary = cv2.erode(binary, erode_kernel)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)

            if area > min_contour_area:
                perimeter = cv2.arcLength(cnt, True)
                hu = cv2.HuMoments(cv2.moments(cnt)).flatten()

                # Ellipse features
                if len(cnt) >= 5:
                    ellipse = cv2.fitEllipse(cnt)
                    (x, y), (major, minor), angle = ellipse
                else:
                    major = minor = angle = -1

                row = [frame_num, time_sec, area, perimeter, major, minor, angle]
                row.extend(hu.tolist())
                writer.writerow(row)

        frame_num += 1

cap.release()
print(f"✅ Done! Features saved to {output_csv}")
