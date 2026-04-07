import cv2
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm


def setup_background_subtractor():
    """Initialize background subtractor for moving object detection"""
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=True
    )
    return backSub


def detect_moving_rat(frame, backSub, frame_count):
    """Detect moving rat using background subtraction"""
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


def process_video_with_motion(video_path, output_csv, output_video,
                              max_minutes=30, target_fps=15,
                              start_minute=0, show_visualization=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_minutes = total_frames / (original_fps * 60)

    print(f"Video: {total_frames} frames, {original_fps} fps, {width}x{height}")
    print(f"Video duration: {total_minutes:.1f} minutes")

    start_frame = int(start_minute * 60 * original_fps)
    if max_minutes:
        end_frame = int((start_minute + max_minutes) * 60 * original_fps)
        end_frame = min(end_frame, total_frames)
    else:
        end_frame = total_frames

    frame_skip = max(1, int(original_fps / target_fps))

    print(f"\nProcessing parameters:")
    print(f"  - Time range: {start_minute}-{start_minute + max_minutes if max_minutes else total_minutes:.1f} minutes")
    print(f"  - Frames: {start_frame}-{end_frame} ({(end_frame - start_frame):,} total)")
    print(f"  - Target FPS: {target_fps} (processing every {frame_skip} frames)")
    print(f"  - Estimated frames to process: {(end_frame - start_frame) // frame_skip:,}")

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

    last_known = {'x': None, 'y': None, 'w': None, 'h': None, 'area': None}

    print(f"\nProcessing video...")

    total_to_process = (end_frame - start_frame) // frame_skip
    pbar = tqdm(total=total_to_process)

    for frame_idx in range(start_frame, end_frame, frame_skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            break

        x, y, w, h, area, fgMask = detect_moving_rat(frame, backSub, frame_idx)

        actual_time_sec = frame_idx / original_fps
        actual_time_min = actual_time_sec / 60

        if x is not None:
            last_known = {'x': x, 'y': y, 'w': w, 'h': h, 'area': area}
            frames_with_detection += 1
        else:
            x    = last_known['x']
            y    = last_known['y']
            w    = last_known['w']
            h    = last_known['h']
            area = last_known['area']

        position_data.append({
            'frame_num': frame_idx,
            'time_sec': actual_time_sec,
            'time_min': actual_time_min,
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'area': area
        })

        frames_processed += 1

        detection_rate = (frames_with_detection / frames_processed) * 100
        pbar.set_description(f"Detections: {frames_with_detection}/{frames_processed} ({detection_rate:.1f}%)")
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
            cv2.putText(combined_frame, "Original + Detection", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined_frame, "Background Subtraction Mask", (width + 10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            out.write(combined_frame)

    pbar.close()

    df = pd.DataFrame(position_data)

    first_det = df[df['x'].notna()].iloc[0] if df['x'].notna().any() else None
    if first_det is not None:
        mask = df.index < df[df['x'].notna()].index[0]
        for col in ('x', 'y', 'width', 'height', 'area'):
            df.loc[mask, col] = first_det[col]

    df.to_csv(output_csv, index=False)

    cap.release()
    if out:
        out.release()

    detection_rate = (frames_with_detection / frames_processed) * 100
    print(f"\n✓ Processing complete!")
    print(f"  - Processed {frames_processed} frames ({frames_processed * frame_skip} total frames considered)")
    print(f"  - Time analyzed: {start_minute}-{actual_time_min:.1f} minutes")
    print(f"  - Movement detections: {frames_with_detection} ({detection_rate:.1f}%)")
    print(f"  - Position data saved to: {output_csv}")
    if show_visualization:
        print(f"  - Visualization saved to: {output_video}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Rat motion detection via background subtraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("output_csv", help="Path for output CSV file")
    parser.add_argument("output_video", help="Path for output visualization video")

    # Configurable parameters
    parser.add_argument("--max-minutes", type=float, default=1,
                        help="Maximum minutes to process (omit or set to 0 for entire video)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Target frames per second to analyze")
    parser.add_argument("--start-minute", type=float, default=0,
                        help="Minute in the video to start processing from")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization output (faster processing)")

    args = parser.parse_args()

    max_minutes = args.max_minutes if args.max_minutes > 0 else None

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    print("Configurable Rat Motion Detection")
    print("=" * 50)

    process_video_with_motion(
        video_path=args.video_path,
        output_csv=args.output_csv,
        output_video=args.output_video,
        max_minutes=max_minutes,
        target_fps=args.fps,
        start_minute=args.start_minute,
        show_visualization=not args.no_viz
    )


if __name__ == "__main__":
    main()
