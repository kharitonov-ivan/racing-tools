#!/usr/bin/env python3
import argparse
import cv2
import sys
from pathlib import Path

# Ensure we can import from overlay
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from overlay import (
        load_session,
        calculate_laps,
        load_track_geometry,
        probe_video,
        REPO_ROOT,
    )
except ImportError as e:
    print(f"Error: Could not import from overlay.py: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"Error during import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Interactive synchronization tool.")
    parser.add_argument("--video", type=Path, required=True, help="Path to video file")
    parser.add_argument("--telemetry", type=Path, required=True, help="Path to telemetry folder")
    parser.add_argument("--track-dir", type=Path, default=REPO_ROOT / "data" / "tracks" / "RIMSportKarting", help="Track directory")
    parser.add_argument("--lap", type=int, default=1, help="Lap number to sync on (default: 1)")
    
    args = parser.parse_args()
    
    if not args.video.is_file():
        sys.exit(f"Video file {args.video} not found")
    if not args.telemetry.is_dir():
        sys.exit(f"Telemetry directory {args.telemetry} not found")

    # 1. Load Telemetry and Find Crossing
    print(f"Loading telemetry from {args.telemetry.name}...")
    # Use a default frequency, we just need the table
    session = load_session(args.telemetry, 20.0, True)
    
    print("Loading track geometry...")
    track_geometry = None
    try:
        track_geometry = load_track_geometry(args.track_dir)
    except Exception as e:
        sys.exit(f"Failed to load track geometry: {e}")
        
    print("Calculating laps to find crossings...")
    session.table, _, crossings = calculate_laps(session.table, track_geometry)
    
    if not crossings:
        sys.exit("No lap crossings found in telemetry.")
        
    crossing_idx = args.lap - 1
    if not (0 <= crossing_idx < len(crossings)):
        sys.exit(f"Lap {args.lap} crossing not found. Total crossings: {len(crossings)}")
        
    telemetry_crossing_time = crossings[crossing_idx]
    print(f"Telemetry crossing time for Lap {args.lap}: {telemetry_crossing_time:.3f}s")
    
    # 2. Open Video
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit("Could not open video.")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video: {fps:.2f} fps, {duration:.2f}s")
    
    # Initial guess: Assume shift is 0, so video time = telemetry time
    # Clamp to video duration
    initial_video_time = max(0.0, min(telemetry_crossing_time, duration))
    current_frame = int(initial_video_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    
    print("\nControls:")
    print("  Left/Right:   -1 / +1 frame")
    print("  Down/Up:      -10 / +10 frames")
    print("  PgDn/PgUp:    -50 / +50 frames")
    print("  Enter:        Confirm Sync")
    print("  Esc:          Cancel")
    
    window_name = "Sync Tool - Find Start/Finish Crossing"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw info on frame
        h, w = frame.shape[:2]
        video_time = current_frame / fps
        
        info_text = [
            f"Frame: {current_frame} / {total_frames}",
            f"Video Time: {video_time:.3f}s",
            f"Telemetry Crossing: {telemetry_crossing_time:.3f}s",
            f"Potential Shift: {telemetry_crossing_time - video_time:.3f}s",
            "Controls: Arrows, PgUp/Dn, Enter to Confirm",
        ]
        
        # Draw background for text
        cv2.rectangle(frame, (10, 10), (600, 150), (0, 0, 0), -1)
        
        y = 40
        for line in info_text:
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y += 30
            
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(0)
        
        if key == 27: # Esc
            print("Cancelled.")
            break
        elif key == 13: # Enter
            print("\nConfirmed!")
            shift = telemetry_crossing_time - video_time
            print(f"Final Time Shift: {shift:.4f}")
            print(f"\nRun overlay with: --time-shift {shift:.4f}")
            break
        elif key == 81 or key == 2: # Left (Linux/Windows codes vary, handling arrows usually requires checking raw codes or using specific libraries)
            # Simple arrow key handling often tricky in cv2.waitKey across platforms
            # Let's try standard codes
            pass
            
        # Arrow keys mapping (common)
        # Left: 81, Up: 82, Right: 83, Down: 84 (on some Linux)
        # Windows: 2424832, 2490368, etc.
        # Let's use WASD or HJKL as backup
        
        if key == ord('d') or key == 83: # Right
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

if __name__ == "__main__":
    main()
