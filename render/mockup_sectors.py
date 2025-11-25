#!/usr/bin/env python3
"""
Mockup script to visualize track segmentation (turns vs straights).
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shapefile

# Add current directory to path to import overlay
sys.path.insert(0, str(Path(__file__).parent.parent))

from render.overlay import load_track_layout, TrackLayout, WGS84_TO_WEBMERC

def calculate_heading(p1, p2):
    """Calculate heading between two points in degrees."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    rads = np.arctan2(dy, dx)
    deg = np.degrees(rads)
    return deg

def normalize_angle(angle):
    """Normalize angle to -180 to 180."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def segment_track(polylines, turn_threshold=2.0):
    """
    Split track into segments based on curvature.
    turn_threshold: degrees of heading change per point to consider a turn.
    """
    # Flatten polylines into a single list of points (assuming single loop)
    # In reality, we might have multiple parts, but usually centerline is one loop
    points = []
    for poly in polylines:
        points.extend(poly)
    
    if not points:
        return []

    # Convert to numpy for easier handling
    points = np.array(points)
    
    # Calculate headings
    headings = []
    for i in range(len(points) - 1):
        h = calculate_heading(points[i], points[i+1])
        headings.append(h)
    
    # Calculate curvature (change in heading)
    curvatures = []
    for i in range(len(headings) - 1):
        diff = normalize_angle(headings[i+1] - headings[i])
        curvatures.append(abs(diff))
    
    # Pad curvatures to match points length
    # First point has no heading, last point has no next point
    # We assign curvature to the point i (representing segment i to i+1)
    curvatures = [0] + curvatures + [0] 
    
    segments = []
    current_type = None # 'straight' or 'turn'
    current_points = []
    
    # Window size for smoothing
    window = 12
    
    for i in range(len(points) - 1):
        # Simple smoothing
        start = max(0, i - window)
        end = min(len(curvatures), i + window + 1)
        avg_curv = np.mean(curvatures[start:end])
        
        segment_type = 'turn' if avg_curv > turn_threshold else 'straight'
        
        if segment_type != current_type:
            if current_points:
                segments.append({"type": current_type, "points": current_points})
            current_type = segment_type
            current_points = [points[i]]
        else:
            current_points.append(points[i])
            
    # Add last segment
    if current_points:
        current_points.append(points[-1])
        segments.append({"type": current_type, "points": current_points})
    
    # Merge small segments
    min_points = 10 # Minimum points to be a valid segment
    
    if len(segments) > 1:
        merged_segments = []
        current_seg = segments[0]
        
        for i in range(1, len(segments)):
            next_seg = segments[i]
            
            # If current segment is too small, merge it into the previous or next?
            # Actually, usually we merge into the *next* if we are iterating?
            # Or we merge into the *larger* neighbour.
            # Simple approach: if segment < min_points, flip its type and merge with neighbours.
            
            # Let's do a multi-pass approach or just a simple filter.
            # If a segment is small, it's likely noise.
            pass
            
        # Re-implement simple merge:
        # 1. Filter out small segments by merging them to the previous one (effectively ignoring the toggle)
        # But we need to be careful about types.
        
        cleaned_segments = []
        if not segments:
            return []
            
        # Start with first
        cleaned_segments.append(segments[0])
        
        for i in range(1, len(segments)):
            seg = segments[i]
            last = cleaned_segments[-1]
            
            if len(seg["points"]) < min_points:
                # Too small, merge into last
                # Note: Type might be different, but we force it to match last to "absorb" the noise
                last["points"].extend(seg["points"])
            else:
                # If type matches last (because we absorbed something), merge
                if seg["type"] == last["type"]:
                    last["points"].extend(seg["points"])
                else:
                    cleaned_segments.append(seg)
        
        segments = cleaned_segments

    return segments

def draw_sectors_mockup():
    width = 1920
    height = 1080
    img = Image.new("RGBA", (width, height), (10, 14, 20, 255))
    draw = ImageDraw.Draw(img)
    
    # Load track
    track_dir = Path("data/tracks/RIMSportKarting")
    try:
        layout = load_track_layout(track_dir)
    except Exception as e:
        print(f"Failed to load track: {e}")
        return

    # Convert to Web Mercator for visualization
    # Note: load_track_layout returns raw coordinates (Web Mercator for this track?)
    # Wait, overlay.py says: 
    # "start_finish_wgs84 = start_finish" (so shapefile is WGS84)
    # But load_track_layout just reads points. 
    # Let's assume centerline is also WGS84 and convert it.
    
    wgs84_polylines = layout.polylines
    webmerc_polylines = []
    for poly in wgs84_polylines:
        lons, lats = zip(*poly)
        
        # Check if already in meters (Web Mercator)
        if abs(lons[0]) > 180 or abs(lats[0]) > 90:
            print("Coordinates appear to be in meters (Web Mercator). Skipping transformation.")
            xs, ys = lons, lats
        else:
            # Transform WGS84 to Web Mercator
            xs, ys = WGS84_TO_WEBMERC.transform(np.array(lons), np.array(lats))
        
        webmerc_polylines.append(list(zip(xs, ys)))
            
    if not webmerc_polylines:
        print("No valid polylines")
        return

    print(f"Processing {sum(len(p) for p in webmerc_polylines)} points across {len(webmerc_polylines)} polylines")
    
    # Segment the track
    # Increased smoothing to 12, threshold to 0.8, and added min_points filter
    segments = segment_track(webmerc_polylines, turn_threshold=0.8)
    
    # Calculate bounds for scaling
    all_points = []
    for seg in segments:
        all_points.extend(seg["points"])
    
    if not all_points:
        print("No points found")
        return
        
    xs, ys = zip(*all_points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    track_w = max_x - min_x
    track_h = max_y - min_y
    
    # Scale to fit screen with padding
    padding = 100
    scale_w = (width - 2 * padding) / track_w
    scale_h = (height - 2 * padding) / track_h
    scale = min(scale_w, scale_h)
    
    # Center
    offset_x = (width - track_w * scale) / 2 - min_x * scale
    offset_y = (height - track_h * scale) / 2 - min_y * scale
    
    # Draw segments
    font = ImageFont.load_default()
    
    for i, seg in enumerate(segments):
        points = seg["points"]
        if len(points) < 2:
            continue
            
        scaled_points = [
            (x * scale + offset_x, height - (y * scale + offset_y)) # Flip Y for screen coords
            for x, y in points
        ]
        
        color = "#5ad2ff" if seg["type"] == "straight" else "#ff7272" # Blue for straight, Red for turn
        width_px = 8 if seg["type"] == "straight" else 8
        
        draw.line(scaled_points, fill=color, width=width_px, joint="curve")
        
        # Label
        mid_idx = len(scaled_points) // 2
        mid_pt = scaled_points[mid_idx]
        label = f"{'S' if seg['type'] == 'straight' else 'T'}{i}"
        
        draw.ellipse((mid_pt[0]-15, mid_pt[1]-15, mid_pt[0]+15, mid_pt[1]+15), fill="black", outline="white")
        
        # Center text roughly
        bbox = draw.textbbox((0,0), label)
        tw = bbox[2]-bbox[0]
        th = bbox[3]-bbox[1]
        draw.text((mid_pt[0]-tw/2, mid_pt[1]-th/2), label, fill="white")

    # Legend
    draw.text((50, 50), "Track Segmentation Mockup", fill="white", font=font)
    draw.rectangle((50, 80, 70, 100), fill="#5ad2ff")
    draw.text((80, 80), "Straight", fill="white", font=font)
    draw.rectangle((50, 110, 70, 130), fill="#ff7272")
    draw.text((80, 110), "Turn", fill="white", font=font)
    
    output_path = Path("render/mockup_sectors.png")
    img.save(output_path)
    print(f"Saved mockup to {output_path}")
    print(f"Found {len(segments)} segments")

if __name__ == "__main__":
    draw_sectors_mockup()
