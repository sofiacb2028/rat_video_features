import cv2
import csv
import time
import argparse
import os

def process_video(video_path, out_csv, target_fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    print(f"Opened: {video_path}")
    print(f"Source FPS: {fps:.2f}, Duration: {duration_sec/3600:.2f} hours, Frames: {total_frames}")

    # Figure out how many frames to skip
    frame_interval = int(round(fps / target_fps))
    print(f"Extracting at {target_fps} fps → every {frame_interval} frames")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["frame_num", "time_sec", "area", "perimeter",
                  "ellipse_major", "ellipse_minor", "ellipse_angle"] + [f"hu{i+1}" for i in range(7)]
        writer.writerow(header)

        frame_num = 0
        frame_read = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_num % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)

                    ellipse_major = ellipse_minor = ellipse_angle = 0
                    if len(cnt) >= 5:
                        ellipse = cv2.fitEllipse(cnt)
                        (center, axes, angle) = ellipse
                        ellipse_major, ellipse_minor = axes
                        ellipse_angle = angle

                    hu = cv2.HuMoments(cv2.moments(cnt)).flatten()

                    time_sec = frame_num / fps
                    writer.writerow([frame_num, time_sec, area, perimeter,
                                     ellipse_major, ellipse_minor, ellipse_angle,
                                     *hu])
                    frame_read += 1

                    if frame_read % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"Processed {frame_read} frames in {elapsed:.1f}s...")

            frame_num += 1

    cap.release()
    elapsed = time.time() - start_time
    print(f"Finished {frame_read} frames in {elapsed:.1f}s. Output saved to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--fps", type=int, required=True, help="Target FPS to extract")
    parser.add_argument("--out", required=True, help="Path to output CSV")
    args = parser.parse_args()

    process_video(args.video, args.out, args.fps)
