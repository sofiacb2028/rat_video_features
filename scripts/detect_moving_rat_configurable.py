import cv2
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def setup_background_subtractor():
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=True
    )
    return backSub


def detect_moving_rat(frame, backSub, frame_count):
    fgMask = backSub.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        min_area = 1000
        max_area = 10000
        area = w * h

        if min_area < area < max_area:
            return (x, y, w, h, area, fgMask)

    return (None, None, None, None, None, fgMask)


def process_video(video_id, video_path, output_csv, output_video,
                  max_minutes=30, target_fps=15,
                  start_minute=0, show_visualization=True):
    print(f"\n[Video {video_id}] Starting: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Video {video_id}] Error: Could not open {video_path}")
        return False

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_minutes = total_frames / (original_fps * 60)

    print(f"[Video {video_id}] {total_frames} frames, {original_fps} fps, {width}x{height}, {total_minutes:.1f} min")

    start_frame = int(start_minute * 60 * original_fps)
    if max_minutes:
        end_frame = int((start_minute + max_minutes) * 60 * original_fps)
        end_frame = min(end_frame, total_frames)
    else:
        end_frame = total_frames

    frame_skip = max(1, int(original_fps / target_fps))

    backSub = setup_background_subtractor()

    if show_visualization:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, target_fps, (width * 2, height))
    else:
        out = None

    position_data = []
    frames_processed = 0
    frames_with_detection = 0
    actual_time_min = start_minute

    total_to_process = (end_frame - start_frame) // frame_skip
    pbar = tqdm(total=total_to_process, desc=f"Video {video_id}", position=video_id)

    for frame_idx in range(start_frame, end_frame, frame_skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            break

        x, y, w, h, area, fgMask = detect_moving_rat(frame, backSub, frame_idx)

        actual_time_sec = frame_idx / original_fps
        actual_time_min = actual_time_sec / 60

        if x is not None:
            frames_with_detection += 1

        position_data.append({
            'video_id':  video_id,
            'frame_num': frame_idx,
            'time_sec':  actual_time_sec,
            'time_min':  actual_time_min,
            'x':         x,
            'y':         y,
            'width':     w,
            'height':    h,
            'area':      area,
        })

        frames_processed += 1
        pbar.update(1)

        if show_visualization and out:
            display_frame = frame.copy()
            if x is not None:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(display_frame, (x + w // 2, y + h // 2), 3, (0, 255, 0), -1)
                cv2.putText(display_frame, f"Frame: {frame_idx} Time: {actual_time_min:.1f}m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Position: ({x},{y}) Area: {area}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, f"No movement - Frame: {frame_idx}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Time: {actual_time_min:.1f} minutes",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            fgMask_display = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
            combined_frame = np.hstack([display_frame, fgMask_display])
            out.write(combined_frame)

    pbar.close()

    df = pd.DataFrame(position_data)

    for col in ('x', 'y', 'width', 'height', 'area'):
        df[col] = (
            pd.to_numeric(df[col], errors='coerce')
            .interpolate(method='linear', limit_direction='both')
            .round()
            .astype('Int64')
        )

    raw_missing = sum(1 for row in position_data if row['x'] is None)
    detection_rate = (frames_with_detection / frames_processed) * 100

    df.to_csv(output_csv, index=False)

    cap.release()
    if out:
        out.release()

    print(f"\n[Video {video_id}] Done — {frames_processed} frames, "
          f"{detection_rate:.1f}% detections, {raw_missing} interpolated. "
          f"Saved to {output_csv}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Rat motion detection — process multiple videos in parallel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("video_paths",   nargs='+', help="One or more input video files")
    parser.add_argument("output_dir",    help="Directory to write output CSVs and videos")
    parser.add_argument("--max-minutes", type=float, default=1,
                        help="Max minutes to process per video (0 = full video)")
    parser.add_argument("--fps",         type=int,   default=15,
                        help="Target frames per second to analyze")
    parser.add_argument("--start-minute",type=float, default=0,
                        help="Minute to start processing from")
    parser.add_argument("--no-viz",      action="store_true",
                        help="Disable visualization output")

    args = parser.parse_args()

    max_minutes = args.max_minutes if args.max_minutes > 0 else None
    os.makedirs(args.output_dir, exist_ok=True)

    jobs = []
    for video_id, video_path in enumerate(args.video_paths):
        stem     = os.path.splitext(os.path.basename(video_path))[0]
        out_csv  = os.path.join(args.output_dir, f"video_{video_id}_{stem}.csv")
        out_vid  = os.path.join(args.output_dir, f"video_{video_id}_{stem}_viz.mp4")
        jobs.append((video_id, video_path, out_csv, out_vid))

    print(f"Processing {len(jobs)} video(s) with {len(jobs)} threads...\n")

    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(
                process_video,
                video_id, video_path, out_csv, out_vid,
                max_minutes, args.fps, args.start_minute, not args.no_viz
            ): video_id
            for video_id, video_path, out_csv, out_vid in jobs
        }

        for future in as_completed(futures):
            video_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[Video {video_id}] Failed: {e}")

    print("\nAll videos processed.")


if __name__ == "__main__":
    main()