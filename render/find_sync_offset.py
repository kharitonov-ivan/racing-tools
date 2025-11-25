
import cv2
import pytesseract
from PIL import Image
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta

def preprocess_image(cv_img):
    # Convert to PIL
    img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    
    # Crop (bottom right)
    w, h = img.size
    crop = img.crop((int(w*0.7), int(h*0.85), w, h))
    
    # Grayscale
    gray = crop.convert('L')
    
    # Threshold
    threshold = 180
    binary = gray.point(lambda p: 255 if p > threshold else 0)
    
    return binary

def parse_time(text):
    # Try to find time pattern HH:MM:SS
    # Sometimes OCR misses colons or reads them as other chars
    text = text.replace(" ", "").replace(".", ":").replace(",", ":")
    
    match = re.search(r'(\d{2}):(\d{2}):(\d{2})', text)
    if match:
        return match.groups()
    
    match = re.search(r'(\d{2}):(\d{2})(\d{2})', text)
    if match:
        return match.groups()
        
    return None

def main():
    video_path = "render/session-01.mp4"
    start_scan = 4.0 # seconds
    duration_scan = 3.0 # seconds
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    
    # Seek to start
    start_frame = int(start_scan * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    prev_time_str = None
    transition_frame_idx = None
    transition_time_str = None
    
    max_frames = int(duration_scan * fps)
    
    print(f"Scanning {max_frames} frames starting at {start_scan}s...")
    
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        processed = preprocess_image(frame)
        
        # OCR
        custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789:/-APM'
        text = pytesseract.image_to_string(processed, config=custom_config)
        
        time_parts = parse_time(text)
        
        if time_parts:
            h, m, s = map(int, time_parts)
            
            # Validation
            if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
                # print(f"Frame {start_frame + i}: Invalid time {h}:{m}:{s}")
                continue
                
            time_str = f"{h:02d}:{m:02d}:{s:02d}"
            
            # print(f"Frame {start_frame + i}: {time_str}")
            
            if prev_time_str and time_str != prev_time_str:
                # Check continuity
                try:
                    prev_h, prev_m, prev_s = map(int, prev_time_str.split(":"))
                    prev_dt = datetime(2025, 1, 1, prev_h, prev_m, prev_s)
                    curr_dt = datetime(2025, 1, 1, h, m, s)
                    diff = (curr_dt - prev_dt).total_seconds()
                    
                    if abs(diff - 1.0) > 0.5:
                        # Not a 1-second increment, likely OCR error or jump
                        # print(f"Frame {start_frame + i}: Time jump detected ({prev_time_str} -> {time_str}), ignoring")
                        continue
                        
                except Exception:
                    pass

                # Transition detected!
                # The current frame (i) is the FIRST frame of the new second.
                transition_frame_idx = start_frame + i
                transition_time_str = time_str
                print(f"\nTransition detected at frame {transition_frame_idx}")
                print(f"Previous time: {prev_time_str}")
                print(f"New time:      {transition_time_str}")
                
                # Calculate exact video time of this frame
                video_time = transition_frame_idx / fps
                print(f"Video Time:    {video_time:.4f} s")
                
                # This video time corresponds to HH:MM:SS.000
                # We need to calculate the offset relative to telemetry start.
                
                # Let's parse the transition time to a datetime object (dummy date)
                t_date = datetime(2025, 1, 1, h, m, s)
                
                # Subtract video time to get video start absolute time
                video_start_abs = t_date - timedelta(seconds=video_time)
                
                print(f"Video Start Absolute Time: {video_start_abs.time()}")
                print(f"Video Start Microseconds: {video_start_abs.microsecond}")
                
                break
                
            prev_time_str = time_str
        else:
            # print(f"Frame {start_frame + i}: No time detected")
            pass
            
    cap.release()

if __name__ == "__main__":
    main()
