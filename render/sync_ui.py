import cv2
import numpy as np

def run_interactive_sync(video_path, crossings, fps=None, duration=None):
    """
    Runs an interactive OpenCV window to synchronize video with telemetry crossings.
    
    Args:
        video_path (Path): Path to video file.
        crossings (list[float]): List of telemetry crossing times (seconds).
        fps (float, optional): Video FPS. If None, probed from video.
        duration (float, optional): Video duration.
        
    Returns:
        float: The calculated time_shift (or None if cancelled).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
        
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if duration is None:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
    total_frames = int(duration * fps)
    
    # State
    current_lap_idx = 0 # Index in crossings list (0 = Lap 1)
    
    # Dictionary to store marked sync points: {lap_idx: video_time}
    marked_points = {}
    
    # Initial seek to first crossing (assuming 0 shift)
    initial_time = max(0.0, min(crossings[0], duration))
    current_frame = int(initial_time * fps)
    
    window_name = "Interactive Sync - Tab: Next Lap | Space: Mark | Enter: Finish"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    final_shift = None
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
            
        video_time = current_frame / fps
        telemetry_time = crossings[current_lap_idx]
        
        # Calculate current potential shift
        current_shift = telemetry_time - video_time
        
        # Prepare Info Text
        info_text = [
            f"Target: Lap {current_lap_idx + 1} Crossing",
            f"Telemetry Time: {telemetry_time:.3f}s",
            f"Video Time:     {video_time:.3f}s",
            f"Current Shift:  {current_shift:.3f}s",
            "",
            f"Marked Laps: {len(marked_points)}",
        ]
        
        if marked_points:
            shifts = []
            for idx, v_time in marked_points.items():
                t_time = crossings[idx]
                s = t_time - v_time
                shifts.append(s)
                mark_str = f"Lap {idx+1}: Shift {s:.3f}s"
                if idx == current_lap_idx:
                    mark_str += " (Current)"
                info_text.append(mark_str)
            
            avg_shift = sum(shifts) / len(shifts)
            drift = max(shifts) - min(shifts) if len(shifts) > 1 else 0.0
            info_text.append("")
            info_text.append(f"Average Shift: {avg_shift:.3f}s")
            info_text.append(f"Max Drift:     {drift:.3f}s")
        else:
            info_text.append("No laps marked yet.")
            
        info_text.append("")
        info_text.append("Controls: Arrows/PgUpDn to Seek | Tab: Change Lap | Space: Mark | Enter: Finish")

        # Draw UI
        # Background
        h, w = frame.shape[:2]
        panel_w = 500
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 40 + len(info_text) * 30), (0, 0, 0), -1)
        
        y = 40
        for line in info_text:
            color = (255, 255, 255)
            if "Average Shift" in line:
                color = (0, 255, 0)
            elif "Target" in line:
                color = (0, 255, 255)
            
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 30
            
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(0)
        
        # Navigation
        if key == 27: # Esc
            print("Cancelled.")
            break
        elif key == 13: # Enter
            if marked_points:
                shifts = [crossings[i] - t for i, t in marked_points.items()]
                final_shift = sum(shifts) / len(shifts)
                print(f"Confirmed! Average Shift: {final_shift:.4f}")
                break
            else:
                print("Please mark at least one lap (Space) before confirming.")
                
        elif key == 9: # Tab
            # Cycle laps
            current_lap_idx = (current_lap_idx + 1) % len(crossings)
            # Seek to approximate location of next lap
            # If we have a known shift (from average), use it. Otherwise assume 0.
            est_shift = 0.0
            if marked_points:
                shifts = [crossings[i] - t for i, t in marked_points.items()]
                est_shift = sum(shifts) / len(shifts)
            
            target_video_time = max(0.0, crossings[current_lap_idx] - est_shift)
            current_frame = int(target_video_time * fps)
            
        elif key == 32: # Space
            # Mark/Unmark
            if current_lap_idx in marked_points:
                del marked_points[current_lap_idx]
            else:
                marked_points[current_lap_idx] = video_time
                
        # Seek controls
        elif key == ord('d') or key == 83: # Right
            current_frame = min(total_frames - 1, current_frame + 1)
        elif key == ord('a') or key == 81: # Left
            current_frame = max(0, current_frame - 1)
        elif key == ord('w') or key == 82: # Up
            current_frame = min(total_frames - 1, current_frame + 10)
        elif key == ord('s') or key == 84: # Down
            current_frame = max(0, current_frame - 10)
        elif key == ord('e') or key == 86: # PgUp
            current_frame = min(total_frames - 1, current_frame + 50)
        elif key == ord('q') or key == 85: # PgDn
            current_frame = max(0, current_frame - 50)
            
    cap.release()
    cv2.destroyAllWindows()
    return final_shift
