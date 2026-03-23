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
    
    if not crossings:
        print("Error: No crossings provided for sync.")
        cap.release()
        return None
    
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
    last_valid_frame = None  # Store last successfully read frame

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            # Show last valid frame instead of closing
            if last_valid_frame is not None:
                frame = last_valid_frame.copy()
            else:
                # No valid frame ever read, exit
                break
        else:
            last_valid_frame = frame.copy()
            
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
            current_frame = min(total_frames - 1, current_frame + 40)
        elif key == ord('s') or key == 84: # Down
            current_frame = max(0, current_frame - 40)
        elif key == ord('e') or key == 86: # PgUp
            current_frame = min(total_frames - 1, current_frame + 50)
        elif key == ord('q') or key == 85: # PgDn
            current_frame = max(0, current_frame - 50)
            
    cap.release()
    cv2.destroyAllWindows()
    return final_shift


def run_manual_lap_marking(video_path, start_time: float = 0.0):
    """
    Runs an interactive OpenCV window to manually mark lap boundaries.
    
    Args:
        video_path (Path): Path to video file.
        start_time (float): Initial time position to start the window at (default 0.0).
        
    Returns:
        list[float]: List of video timestamps for lap boundaries (start of Lap 1, start of Lap 2, etc.)
                     Returns None if cancelled.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
        
    total_frames = int(duration * fps)
    frames_7s = int(7 * fps)
    
    # State
    # List of marked timestamps (video time)
    # We keep them sorted
    marked_boundaries = []
    
    # Start at specified time
    current_frame = int(start_time * fps)
    
    window_name = "Manual Lap Marking - Space: Mark/Unmark | Enter: Finish"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    result = None
    last_valid_frame = None  # Store last successfully read frame

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            # Show last valid frame instead of closing
            if last_valid_frame is not None:
                frame = last_valid_frame.copy()
            else:
                # No valid frame ever read, exit
                break
        else:
            last_valid_frame = frame.copy()
            
        video_time = current_frame / fps
        
        # Prepare Info Text
        info_text = [
            f"Video Time: {video_time:.3f}s",
            "",
            f"Marked Boundaries: {len(marked_boundaries)}",
        ]
        
        # Show last few boundaries
        sorted_boundaries = sorted(marked_boundaries)
        for i, t in enumerate(sorted_boundaries):
            # Show Lap N (Interval between t_i and t_{i+1})
            # Boundary i starts Lap i+1 (if we assume Lap 0 is before first mark)
            # Or usually: First mark is Start/Finish line crossing starting Lap 1 (End of Out Lap)
            # So:
            # < Mark 0: Out Lap (Lap 0)
            # Mark 0 - Mark 1: Lap 1
            # Mark 1 - Mark 2: Lap 2
            
            label = f"Start Lap {i+1}"
            info_text.append(f"{label}: {t:.3f}s")
            
        info_text.append("")
        info_text.append("Controls:")
        info_text.append("  Arrows : Seek (Left/Right: 1fr, Up/Down: 40fr)")
        info_text.append("  PgUp/Dn: Seek 50fr")
        info_text.append("  z / x  : Seek -/+ 7s")
        info_text.append("  Space  : Mark/Unmark")
        info_text.append("  Enter  : Finish")

        # Draw UI
        # Background
        h, w = frame.shape[:2]
        panel_w = 500
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 40 + len(info_text) * 30), (0, 0, 0), -1)
        
        y = 40
        for line in info_text:
            color = (255, 255, 255)
            if "Video Time" in line:
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
            if marked_boundaries:
                result = sorted(marked_boundaries)
                print(f"Confirmed {len(result)} boundaries.")
                break
            else:
                print("Please mark at least one boundary before confirming.")
                
        elif key == 32: # Space
            # Mark/Unmark current time
            # Check if close to an existing mark (within 0.5s), if so remove it
            # Else add new
            
            # Simple exact match is unlikely, use threshold
            threshold = 0.5
            found_idx = -1
            for i, t in enumerate(marked_boundaries):
                if abs(t - video_time) < threshold:
                    found_idx = i
                    break
            
            if found_idx >= 0:
                print(f"Removed mark at {marked_boundaries[found_idx]:.3f}s")
                del marked_boundaries[found_idx]
            else:
                marked_boundaries.append(video_time)
                print(f"Added mark at {video_time:.3f}s")
                
        # Seek controls
        elif key == ord('z'): # Seek -7s
            current_frame = max(0, current_frame - frames_7s)
        elif key == ord('x'): # Seek +7s
            current_frame = min(total_frames - 1, current_frame + frames_7s)
        elif key == ord('d') or key == 83: # Right
            current_frame = min(total_frames - 1, current_frame + 1)
        elif key == ord('a') or key == 81: # Left
            current_frame = max(0, current_frame - 1)
        elif key == ord('w') or key == 82: # Up
            current_frame = min(total_frames - 1, current_frame + 40)
        elif key == ord('s') or key == 84: # Down
            current_frame = max(0, current_frame - 40)
        elif key == ord('e') or key == 86: # PgUp
            current_frame = min(total_frames - 1, current_frame + 50)
        elif key == ord('q') or key == 85: # PgDn
            current_frame = max(0, current_frame - 50)
            
    cap.release()
    cv2.destroyAllWindows()
    return result

def run_trim_selection(video_path):
    """
    Interactive UI to select start and end points for trimming.
    Returns (start_time, end_time) in seconds.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    frames_7s = int(7 * fps)

    current_frame = 0
    start_time = 0.0
    end_time = duration
    
    # Flags to indicate if start/end have been explicitly set by user
    start_set = False
    end_set = False
    confirming = False
    confirmed = False

    window_name = "Trim Selection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            # If we fail to read, it might be because we are at the end or frame count is wrong.
            # Try to step back aggressively to find the last valid frame
            print(f"Warning: Could not read frame {current_frame}. Backtracking...")
            found = False
            for _ in range(50): # Try backing off up to 50 frames
                if current_frame > 0:
                    current_frame -= 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                    ret, frame = cap.read()
                    if ret:
                        found = True
                        break
                else:
                    break
            
            if not found:
                print("Error: Could not read any valid frame near this position.")
                break

        # Overlay Info
        current_time = current_frame / fps
        
        # Helper text
        text_lines = [
            f"Time: {current_time:.3f}s / {duration:.3f}s (Frame {current_frame}/{total_frames})",
            f"Start (In): {start_time:.3f}s {'[SET]' if start_set else '[DEFAULT]'}",
            f"End (Out):  {end_time:.3f}s {'[SET]' if end_set else '[DEFAULT]'}",
            f"Duration:   {end_time - start_time:.3f}s",
            "",
            "Controls:",
            "  [ or i : Set Start (In) point",
            "  ] or o : Set End (Out) point",
            "  r      : Reset selection",
            "  Space  : Play/Pause (hold to play)",
            f"  Arrows : Seek (Left/Right: 1fr, Up/Down: 40fr)",
            f"  PgUp/Dn: Seek 50fr",
            f"  z / x  : Seek -/+ 7s",
            f"  g / h  : Jump to Start",
            f"  G / t  : Jump to End",
            f"  Enter  : Confirm Trim",
            f"  Esc    : Cancel"
        ]

        # Draw text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 520), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = 40
        for line in text_lines:
            color = (255, 255, 255)
            if "Start (In)" in line and start_set: color = (0, 255, 0)
            if "End (Out)" in line and end_set: color = (0, 0, 255)
            
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 30

        if confirming:
            # Darken screen more
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            # Show confirmation message
            cv2.putText(frame, "CONFIRM TRIM?", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
            cv2.putText(frame, f"Start: {start_time:.2f}s", (450, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"End:   {end_time:.2f}s", (450, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'y' to proceed, 'n' to cancel", (350, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(0)

        if confirming:
            if key == ord('y') or key == 13: # y or Enter again
                confirmed = True
                break
            elif key == ord('n') or key == 27: # n or Esc
                confirming = False
            continue

        if key == 27: # Esc
            cap.release()
            cv2.destroyAllWindows()
            return None, None
        
        elif key == 13: # Enter
            confirming = True
            
        elif key == ord('[') or key == ord('i'): # Set Start
            start_time = current_time
            start_set = True
            # Ensure start < end
            if start_time >= end_time:
                end_time = duration
                end_set = False
                
        elif key == ord(']') or key == ord('o'): # Set End
            end_time = current_time
            end_set = True
            # Ensure end > start
            if end_time <= start_time:
                start_time = 0.0
                start_set = False
                
        elif key == ord('r'): # Reset
            start_time = 0.0
            end_time = duration
            start_set = False
            end_set = False

        # Navigation
        elif key == ord('z'): # Seek -7s
            current_frame = max(0, current_frame - frames_7s)
        elif key == ord('x'): # Seek +7s
            current_frame = min(total_frames - 1, current_frame + frames_7s)
        elif key == ord('g') or key == ord('h'): # Jump to Start
            current_frame = 0
        elif key == ord('G') or key == ord('t'): # Jump to End
            current_frame = total_frames - 1
        elif key == ord('a') or key == 81: # Left
            current_frame = max(0, current_frame - 1)
        elif key == ord('d') or key == 83: # Right
            current_frame = min(total_frames - 1, current_frame + 1)
        elif key == ord('w') or key == 82: # Up
            current_frame = min(total_frames - 1, current_frame + 40)
        elif key == ord('s') or key == 84: # Down
            current_frame = max(0, current_frame - 40)
        elif key == ord('e') or key == 86: # PgUp
            current_frame = min(total_frames - 1, current_frame + 50)
        elif key == ord('q') or key == 85: # PgDn
            current_frame = max(0, current_frame - 50)
        elif key == 32: # Space (advance 1 frame, acting as play if held, though waitKey(0) pauses)
             current_frame = min(total_frames - 1, current_frame + 1)

    cap.release()
    cv2.destroyAllWindows()
    
    if confirmed:
        return start_time, end_time
    else:
        return None, None
