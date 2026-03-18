import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def setup_background_subtractor():
    """Initialize background subtractor for moving object detection"""
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=500,        # How many frames to learn background
        varThreshold=16,    # Lower = more sensitive
        detectShadows=True  # Detect but mark shadows differently
    )
    return backSub

def detect_moving_rat(frame, backSub, frame_count):
    """
    Detect moving rat using background subtraction
    """
    # Apply background subtraction
    fgMask = backSub.apply(frame)
    
    # Noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #originally 3,3
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

    # ADD MORPHOLOGY CLOSING TO FILL GAPS
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assuming it's the moving rat)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Filter by size - adjust these based on your rat size!
        min_area = 1000   # pixels - rat should be larger than noise
        max_area = 10000  # pixels - rat shouldn't be huge
        area = w * h
        
        if min_area < area < max_area:
            return (x, y, w, h, area, fgMask)
    
    return (None, None, None, None, None, fgMask)

def process_video_with_motion(video_path, output_csv, output_video, 
                            max_minutes=30, target_fps=15, 
                            start_minute=0, show_visualization=True):
    """
    Process video to detect MOVING rat positions with configurable parameters
    
    Args:
        video_path: Path to input video
        output_csv: Path for output CSV file
        output_video: Path for output visualization video
        max_minutes: Maximum minutes to process (None for entire video)
        target_fps: Frames per second to analyze (15 fps = process every 2nd frame at 30 fps)
        start_minute: Minute to start processing from
        show_visualization: Whether to create output video (slower but visual)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_minutes = total_frames / (original_fps * 60)
    
    print(f"Video: {total_frames} frames, {original_fps} fps, {width}x{height}")
    print(f"Video duration: {total_minutes:.1f} minutes")
    
    # Calculate frame range based on time parameters
    start_frame = int(start_minute * 60 * original_fps)
    if max_minutes:
        end_frame = int((start_minute + max_minutes) * 60 * original_fps)
        end_frame = min(end_frame, total_frames)
    else:
        end_frame = total_frames
    
    # Calculate frame skip for target FPS
    frame_skip = max(1, int(original_fps / target_fps))
    
    print(f"\nProcessing parameters:")
    print(f"  - Time range: {start_minute}-{start_minute + max_minutes if max_minutes else total_minutes:.1f} minutes")
    print(f"  - Frames: {start_frame}-{end_frame} ({(end_frame - start_frame):,} total)")
    print(f"  - Target FPS: {target_fps} (processing every {frame_skip} frames)")
    print(f"  - Estimated frames to process: {(end_frame - start_frame) // frame_skip:,}")
    
    # Setup background subtractor
    backSub = setup_background_subtractor()
    
    # Setup output video if requested
    if show_visualization:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_width = width * 2 if show_visualization else width
        out = cv2.VideoWriter(output_video, fourcc, target_fps, (out_width, height))
    else:
        out = None
    
    position_data = []
    frames_processed = 0
    frames_with_detection = 0
    
    print(f"\nProcessing video...")
    
    # Create progress bar
    total_to_process = (end_frame - start_frame) // frame_skip
    pbar = tqdm(total=total_to_process)
    
    for frame_idx in range(start_frame, end_frame, frame_skip):
        # Set video position to current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detect moving objects
        x, y, w, h, area, fgMask = detect_moving_rat(frame, backSub, frame_idx)
        
        # Calculate actual time in video
        actual_time_sec = frame_idx / original_fps
        actual_time_min = actual_time_sec / 60
        
        # Store data
        detection_data = {
            'frame_num': frame_idx,
            'time_sec': actual_time_sec,
            'time_min': actual_time_min,
            'x': x,
            'y': y, 
            'width': w,
            'height': h,
            'area': area,
            'detection': x is not None
        }
        position_data.append(detection_data)
        
        if x is not None:
            frames_with_detection += 1
        
        frames_processed += 1
        
        # Update progress bar with detection rate
        detection_rate = (frames_with_detection / frames_processed) * 100
        pbar.set_description(f"Detections: {frames_with_detection}/{frames_processed} ({detection_rate:.1f}%)")
        pbar.update(1)
        
        # Create visualization if requested
        if show_visualization and out:
            display_frame = frame.copy()
            
            if x is not None:
                # Draw bounding box on original frame
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Add center point
                center_x, center_y = x + w//2, y + h//2
                cv2.circle(display_frame, (center_x, center_y), 3, (0, 255, 0), -1)
                
                # Add info text
                info_text = f"Frame: {frame_idx} Time: {actual_time_min:.1f}m"
                cv2.putText(display_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Position: ({x},{y}) Area: {area}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # No detection text
                cv2.putText(display_frame, f"No movement - Frame: {frame_idx}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Time: {actual_time_min:.1f} minutes", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Enhance mask for display
            fgMask_display = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
            
            # Combine original frame with mask view
            combined_frame = np.hstack([display_frame, fgMask_display])
            
            # Add labels
            cv2.putText(combined_frame, "Original + Detection", (10, height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined_frame, "Background Subtraction Mask", (width + 10, height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write combined frame
            out.write(combined_frame)
    
    pbar.close()
    
    # Save data
    df = pd.DataFrame(position_data)
    df.to_csv(output_csv, index=False)
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    
    # Statistics
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
    video_path = "/home/sbcubillan-bravo/data/bottom_left.mp4"
    output_csv = "/home/sbcubillan-bravo/rat_behavior_analysis/results/motion_position_data2.csv"
    output_video = "/home/sbcubillan-bravo/rat_behavior_analysis/results/motion_tracking2.mp4"
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # CONFIGURE THESE PARAMETERS:
    MAX_MINUTES = 3        # Process first 30 minutes (set to None for entire video)
    TARGET_FPS = 15         # Analyze 15 frames per second
    START_MINUTE = 25        # Start from beginning (change if you want later segment)
    SHOW_VIZ = True         # Set to False for faster processing without video output
    
    print("🎯 Configurable Rat Motion Detection")
    print("=" * 50)
    
    process_video_with_motion(
        video_path=video_path,
        output_csv=output_csv,
        output_video=output_video,
        max_minutes=MAX_MINUTES,
        target_fps=TARGET_FPS,
        start_minute=START_MINUTE,
        show_visualization=SHOW_VIZ
    )

if __name__ == "__main__":
    main()
