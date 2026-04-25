import cv2
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def setup_background_subtractor():
    """Create a fresh background subtractor with tuned parameters for noise reduction"""
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=300,           # Reduced history for faster adaptation
        varThreshold=25,       # Increased threshold to reduce noise sensitivity
        detectShadows=True     # Keep shadow detection on
    )
    return backSub


def create_ceiling_mask(height, width):
    """Create a mask that blocks out the top quarter of the frame (ceiling)"""
    mask = np.ones((height, width), dtype=np.uint8) * 255
    ceiling_height = height // 4
    mask[0:ceiling_height, :] = 0  # Black out top quarter
    return mask


def detect_movement_mask(frame, backSub, ceiling_mask):
    """
    Apply background subtraction and return binary movement mask.
    Filters out ceiling area and small noise.
    """
    fgMask = backSub.apply(frame, learningRate=0.01)  # Slower learning rate for stability
    
    # Apply ceiling mask to ignore top quarter
    fgMask = cv2.bitwise_and(fgMask, ceiling_mask)
    
    # Clean up mask with morphological operations
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Remove small noise
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel_small)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_medium)
    
    # Find all contours
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rat_detected = False
    rat_bbox = None
    filtered_mask = np.zeros_like(fgMask)  # Mask that only contains rat-sized movement
    
    if contours:
        # Filter contours by area
        min_area = 300
        max_area = 5000
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                valid_contours.append(contour)
        
        if valid_contours:
            # Use the largest valid contour as the rat
            largest_contour = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            area = w * h
            
            rat_detected = True
            rat_bbox = (x, y, w, h, area)
            
            # Create filtered mask - ONLY movement inside the rat bounding box
            cv2.rectangle(filtered_mask, (x, y), (x + w, y + h), 255, -1)
            filtered_mask = cv2.bitwise_and(fgMask, filtered_mask)
    
    return fgMask, filtered_mask, rat_detected, rat_bbox


def process_epoch(cap, epoch_idx, start_time_sec, original_fps, width, height, ceiling_mask):
    """
    Process a single 10-second epoch, sampling 1 frame per second.
    Returns accumulated mask, epoch data, and list of frames for visualization.
    """
    EPOCH_DURATION = 10  # seconds
    FRAMES_PER_EPOCH = 10  # 1 frame per second
    
    # Fresh background subtractor for this epoch
    backSub = setup_background_subtractor()
    
    # Initialize accumulating masks (grayscale, starts at 0/black)
    accumulated_mask_all = np.zeros((height, width), dtype=np.uint8)
    accumulated_mask_filtered = np.zeros((height, width), dtype=np.uint8)
    
    epoch_data = {
        'epoch_id': epoch_idx,
        'start_time_sec': start_time_sec,
        'end_time_sec': start_time_sec + EPOCH_DURATION,
        'frames_processed': 0,
        'frames_with_rat': 0,
        'cumulative_movement_pixels_all': 0,
        'cumulative_movement_pixels_filtered': 0,
        'final_mask_white_pixels': 0,
        'rat_bboxes': []
    }
    
    viz_frames = []  # Store (frame, accumulated_mask, filtered_mask) for video output
    
    # Sample exactly 1 frame per second
    for second_offset in range(EPOCH_DURATION):
        frame_time_sec = start_time_sec + second_offset
        frame_idx = int(frame_time_sec * original_fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Get movement masks for this frame
        fgMask, filtered_mask, rat_detected, rat_bbox = detect_movement_mask(frame, backSub, ceiling_mask)
        
        # Count white pixels in masks
        white_pixels_all = np.count_nonzero(fgMask)
        white_pixels_filtered = np.count_nonzero(filtered_mask)
        
        epoch_data['cumulative_movement_pixels_all'] += white_pixels_all
        epoch_data['cumulative_movement_pixels_filtered'] += white_pixels_filtered
        
        if rat_detected:
            epoch_data['frames_with_rat'] += 1
            epoch_data['rat_bboxes'].append(rat_bbox)
            
            # Add to filtered accumulating mask only when rat is detected
            # Add 25 intensity per frame where movement is detected
            add_mask = np.where(filtered_mask > 0, 25, 0).astype(np.uint8)
            accumulated_mask_filtered = cv2.add(accumulated_mask_filtered, add_mask)
        else:
            # Still store a null bbox for consistent length if needed
            epoch_data['rat_bboxes'].append(None)
        
        # Add to all-movement accumulating mask (for comparison visualization)
        add_mask_all = np.where(fgMask > 0, 25, 0).astype(np.uint8)
        accumulated_mask_all = cv2.add(accumulated_mask_all, add_mask_all)
        
        epoch_data['frames_processed'] += 1
        
        # Store for visualization
        viz_frames.append((
            frame.copy(), 
            accumulated_mask_filtered.copy(), 
            accumulated_mask_all.copy(),
            second_offset, 
            rat_detected, 
            rat_bbox
        ))
    
    # Final metrics - use filtered mask as the primary output
    epoch_data['final_mask_white_pixels'] = np.count_nonzero(accumulated_mask_filtered)
    
    # Filter out None values from rat_bboxes before calculating averages
    valid_bboxes = [b for b in epoch_data['rat_bboxes'] if b is not None]
    epoch_data['valid_rat_bboxes'] = valid_bboxes
    
    return accumulated_mask_filtered, epoch_data, viz_frames


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
    total_seconds = total_frames / original_fps
    
    print(f"[Video {video_id}] {total_frames} frames, {original_fps} fps, {width}x{height}, {total_seconds/60:.1f} min")
    
    # Create ceiling mask once (ignore top quarter of frame)
    ceiling_mask = create_ceiling_mask(height, width)
    
    # Calculate processing range
    start_second = start_minute * 60
    if max_minutes:
        end_second = start_second + (max_minutes * 60)
        end_second = min(end_second, total_seconds)
    else:
        end_second = total_seconds
    
    EPOCH_DURATION = 10  # seconds
    
    # Calculate number of complete epochs
    num_epochs = int((end_second - start_second) // EPOCH_DURATION)
    
    print(f"[Video {video_id}] Processing {num_epochs} epochs ({start_minute:.1f}-{end_second/60:.1f} min)")
    print(f"[Video {video_id}] Ignoring top {height//4} pixels (ceiling)")
    
    # Setup video writer if visualization enabled
    if show_visualization:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Three panels: Original | All Movement | Filtered Movement
        out = cv2.VideoWriter(output_video, fourcc, 1.0, (width * 3, height))
    else:
        out = None
    
    all_epoch_data = []
    
    pbar = tqdm(total=num_epochs, desc=f"Video {video_id}", position=video_id)
    
    for epoch_idx in range(num_epochs):
        epoch_start_sec = start_second + (epoch_idx * EPOCH_DURATION)
        
        # Process this epoch
        accumulated_mask, epoch_data, viz_frames = process_epoch(
            cap, epoch_idx, epoch_start_sec, original_fps, width, height, ceiling_mask
        )
        
        all_epoch_data.append(epoch_data)
        
        # Write visualization frames for this epoch
        if show_visualization and out:
            for frame, acc_mask_filtered, acc_mask_all, second_offset, rat_detected, rat_bbox in viz_frames:
                current_time_min = (epoch_start_sec + second_offset) / 60
                
                # === Panel 1: Original frame with annotations ===
                display_frame = frame.copy()
                
                # Draw ceiling line
                ceiling_y = height // 4
                cv2.line(display_frame, (0, ceiling_y), (width, ceiling_y), (255, 255, 0), 1)
                cv2.putText(display_frame, "CEILING (ignored)", 
                            (10, ceiling_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                if rat_detected and rat_bbox:
                    x, y, w, h, area = rat_bbox
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(display_frame, (x + w // 2, y + h // 2), 3, (0, 255, 0), -1)
                    cv2.putText(display_frame, f"RAT - Area: {area} px", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "No rat detected", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                cv2.putText(display_frame, f"Epoch {epoch_idx} | Time: {current_time_min:.1f} min", 
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Sec {second_offset+1}/10", 
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # === Panel 2: All movement mask ===
                mask_all_display = cv2.cvtColor(acc_mask_all, cv2.COLOR_GRAY2BGR)
                white_all = np.count_nonzero(acc_mask_all)
                cv2.putText(mask_all_display, "ALL MOVEMENT", 
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(mask_all_display, f"Pixels: {white_all}", 
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # === Panel 3: Filtered movement mask (rat only) ===
                mask_filtered_display = cv2.cvtColor(acc_mask_filtered, cv2.COLOR_GRAY2BGR)
                white_filtered = np.count_nonzero(acc_mask_filtered)
                cv2.putText(mask_filtered_display, "RAT MOVEMENT ONLY", 
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(mask_filtered_display, f"Pixels: {white_filtered}", 
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Combine all three panels
                combined_frame = np.hstack([display_frame, mask_all_display, mask_filtered_display])
                out.write(combined_frame)
        
        pbar.update(1)
        pbar.set_postfix({
            'rat_frames': epoch_data['frames_with_rat'],
            'filt_px': epoch_data['final_mask_white_pixels']
        })
    
    pbar.close()
    
    # Create DataFrame from epoch data
    df = pd.DataFrame(all_epoch_data)
    
    # Calculate average bounding box for epochs with rat detections
    df['avg_rat_x'] = df['valid_rat_bboxes'].apply(
        lambda bboxes: int(np.mean([b[0] for b in bboxes])) if bboxes else None
    )
    df['avg_rat_y'] = df['valid_rat_bboxes'].apply(
        lambda bboxes: int(np.mean([b[1] for b in bboxes])) if bboxes else None
    )
    df['avg_rat_width'] = df['valid_rat_bboxes'].apply(
        lambda bboxes: int(np.mean([b[2] for b in bboxes])) if bboxes else None
    )
    df['avg_rat_height'] = df['valid_rat_bboxes'].apply(
        lambda bboxes: int(np.mean([b[3] for b in bboxes])) if bboxes else None
    )
    df['avg_rat_area'] = df['valid_rat_bboxes'].apply(
        lambda bboxes: int(np.mean([b[4] for b in bboxes])) if bboxes else None
    )
    
    # Keep only the columns we want in final CSV
    output_columns = [
        'epoch_id', 'start_time_sec', 'end_time_sec',
        'frames_processed', 'frames_with_rat',
        'cumulative_movement_pixels_all',
        'cumulative_movement_pixels_filtered',
        'final_mask_white_pixels',
        'avg_rat_x', 'avg_rat_y', 'avg_rat_width', 'avg_rat_height', 'avg_rat_area'
    ]
    
    df_output = df[output_columns].copy()
    
    # Save CSV
    df_output.to_csv(output_csv, index=False)
    
    # Print summary
    total_rat_frames = df['frames_with_rat'].sum()
    total_frames_processed = df['frames_processed'].sum()
    detection_rate = (total_rat_frames / total_frames_processed * 100) if total_frames_processed > 0 else 0
    
    print(f"\n[Video {video_id}] Done — {num_epochs} epochs, "
          f"{detection_rate:.1f}% frames with rat detection.")
    print(f"[Video {video_id}] CSV saved to: {output_csv}")
    if show_visualization:
        print(f"[Video {video_id}] Video saved to: {output_video}")
    
    cap.release()
    if out:
        out.release()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Rat motion detection — epoch-based with noise filtering and ceiling ignore",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("video_paths",   nargs='+', help="One or more input video files")
    parser.add_argument("output_dir",    help="Directory to write output CSVs and videos")
    parser.add_argument("--max-minutes", type=float, default=1,
                        help="Max minutes to process per video (0 = full video)")
    parser.add_argument("--fps",         type=int,   default=15,
                        help="Target frames per second (used for reference)")
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
        out_csv  = os.path.join(args.output_dir, f"video_{video_id}_{stem}_epochs.csv")
        out_vid  = os.path.join(args.output_dir, f"video_{video_id}_{stem}_epochs_viz.mp4")
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
