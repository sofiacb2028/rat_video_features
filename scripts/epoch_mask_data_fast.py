import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


EPOCH_DURATION_SEC = 10


def setup_background_subtractor():
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=True
    )
    return backSub


def create_ceiling_mask(height, width):
    mask = np.ones((height, width), dtype=np.uint8) * 255
    mask[0:height // 4, :] = 0
    return mask


def save_epoch_array(mask_list, epoch_idx, video_id, output_npy_dir):
    """
    Stack binary masks, compute mean across frames → float32 array in [0.0, 1.0].
    Shape: (H, W). A value of 1.0 means that pixel was active in every frame.
    """
    stack = np.stack(mask_list, axis=0)
    epoch_array = stack.mean(axis=0).astype(np.float32)

    filename = f"video_{video_id}_epoch_{epoch_idx:04d}.npy"
    path = os.path.join(output_npy_dir, filename)
    np.save(path, epoch_array)


def process_video(video_id, video_path, output_npy_dir,
                  max_minutes=None, target_fps=5, start_minute=0,
                  scale=0.5):
    print(f"\n[Video {video_id}] Starting: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Video {video_id}] Error: Could not open {video_path}")
        return False

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_minutes = total_frames / (original_fps * 60)

    # Resized dimensions
    width  = int(orig_width  * scale)
    height = int(orig_height * scale)

    print(f"[Video {video_id}] {total_frames} frames, {original_fps} fps, "
          f"{orig_width}x{orig_height} → resized to {width}x{height}, "
          f"{total_minutes:.1f} min")

    start_frame = int(start_minute * 60 * original_fps)
    if max_minutes:
        end_frame = int((start_minute + max_minutes) * 60 * original_fps)
        end_frame = min(end_frame, total_frames)
    else:
        end_frame = total_frames

    # How many raw frames to skip to achieve target_fps
    frame_skip = max(1, round(original_fps / target_fps))

    # How many sampled frames fit in one epoch
    frames_per_epoch = int(EPOCH_DURATION_SEC * original_fps / frame_skip)

    # Ceiling mask built at resized resolution
    ceiling_mask = create_ceiling_mask(height, width)

    # Single background subtractor — never reset, persists across entire video
    backSub = setup_background_subtractor()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    epoch_idx = 0
    current_epoch_masks = []

    # Seek to start frame once, then read sequentially
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    total_to_process = (end_frame - start_frame) // frame_skip
    pbar = tqdm(total=total_to_process, desc=f"Video {video_id}", position=video_id)

    raw_frame_idx = start_frame

    while raw_frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process frames that land on our target fps interval
        if (raw_frame_idx - start_frame) % frame_skip == 0:

            # Resize for speed
            frame = cv2.resize(frame, (width, height))

            # Background subtraction
            fgMask = backSub.apply(frame)

            # Apply ceiling mask
            fgMask = cv2.bitwise_and(fgMask, ceiling_mask)

            # Morphological cleanup
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)

            # Binary mask: 1 where movement, 0 elsewhere
            binary_mask = (fgMask > 0).astype(np.uint8)
            current_epoch_masks.append(binary_mask)

            # Flush complete epoch to disk
            if len(current_epoch_masks) >= frames_per_epoch:
                save_epoch_array(current_epoch_masks, epoch_idx, video_id, output_npy_dir)
                epoch_idx += 1
                current_epoch_masks = []

            pbar.update(1)

        raw_frame_idx += 1

    # Save any remaining frames as a partial final epoch
    if current_epoch_masks:
        print(f"\n[Video {video_id}] Saving partial epoch {epoch_idx} "
              f"({len(current_epoch_masks)}/{frames_per_epoch} frames)")
        save_epoch_array(current_epoch_masks, epoch_idx, video_id, output_npy_dir)

    pbar.close()
    cap.release()

    print(f"[Video {video_id}] Done — {epoch_idx + 1} epochs saved to: {output_npy_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Rat motion detection — saves per-epoch normalized movement arrays for sleep staging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("video_paths",    nargs='+', help="One or more input video files")
    parser.add_argument("output_dir",     help="Directory to write epoch .npy files")
    parser.add_argument("--max-minutes",  type=float, default=0,
                        help="Max minutes to process per video (0 = full video)")
    parser.add_argument("--fps",          type=int,   default=5,
                        help="Target frames per second to analyze")
    parser.add_argument("--start-minute", type=float, default=0,
                        help="Minute to start processing from")
    parser.add_argument("--scale",        type=float, default=0.5,
                        help="Frame resize scale factor (0.5 = half resolution)")

    args = parser.parse_args()

    max_minutes = args.max_minutes if args.max_minutes > 0 else None
    os.makedirs(args.output_dir, exist_ok=True)

    jobs = []
    for video_id, video_path in enumerate(args.video_paths):
        stem        = os.path.splitext(os.path.basename(video_path))[0]
        out_npy_dir = os.path.join(args.output_dir, f"video_{video_id}_{stem}_epochs")
        os.makedirs(out_npy_dir, exist_ok=True)
        jobs.append((video_id, video_path, out_npy_dir))

    print(f"Processing {len(jobs)} video(s) with {len(jobs)} threads...\n")

    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(
                process_video,
                video_id, video_path, out_npy_dir,
                max_minutes, args.fps, args.start_minute, args.scale
            ): video_id
            for video_id, video_path, out_npy_dir in jobs
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
