#!/usr/bin/env python3
"""
Mockup script to generate a static overlay example without running the full pipeline.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def ensure_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a font."""
    try:
        # Try to find a nice font
        font_names = ["Roboto-Bold.ttf" if bold else "Roboto-Regular.ttf", 
                      "Arial Bold.ttf" if bold else "Arial.ttf",
                      "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
        
        font_path = None
        # Check common system paths (simplified for mockup)
        search_paths = [
            Path("/usr/share/fonts"),
            Path("/mnt/c/Windows/Fonts"),
        ]
        
        for path in search_paths:
            for name in font_names:
                found = list(path.rglob(name))
                if found:
                    font_path = found[0]
                    break
            if font_path:
                break
                
        if font_path:
            return ImageFont.truetype(str(font_path), size)
    except Exception:
        pass
    return ImageFont.load_default()

def draw_grid(draw, width, height, step=100):
    """Draw a layout grid."""
    for x in range(0, width, step):
        draw.line([(x, 0), (x, height)], fill=(255, 255, 255, 30), width=1)
        draw.text((x + 2, 2), str(x), fill=(255, 255, 255, 100))
    for y in range(0, height, step):
        draw.line([(0, y), (width, y)], fill=(255, 255, 255, 30), width=1)
        draw.text((2, y + 2), str(y), fill=(255, 255, 255, 100))
    
    # Center lines
    draw.line([(width/2, 0), (width/2, height)], fill=(255, 0, 0, 100), width=2)
    draw.line([(0, height/2), (width, height/2)], fill=(255, 0, 0, 100), width=2)

def draw_mockup(debug=True):
    width = 1920
    height = 1080
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0)) # Transparent background
    draw = ImageDraw.Draw(img)

    if debug:
        draw_grid(draw, width, height)

    # --- Layout Configuration ---
    # Bottom Left: Track Map
    # Bottom Center: Speed/RPM/Gear
    # Bottom Right: Lap Time / Best Lap
    
    # 1. Track Map (Bottom Left)
    map_size = 300
    map_x = 60
    map_y = height - map_size - 60
    
    # Background panel
    draw.rounded_rectangle(
        (map_x, map_y, map_x + map_size, map_y + map_size),
        radius=20,
        fill=(10, 14, 20, 200),
        outline="#214d66",
        width=2
    )
    
    # Dummy Track
    track_points = [
        (0.2, 0.8), (0.2, 0.2), (0.5, 0.1), (0.8, 0.2), (0.8, 0.8), (0.5, 0.9), (0.2, 0.8)
    ]
    scaled_points = [
        (map_x + p[0] * map_size, map_y + p[1] * map_size) for p in track_points
    ]
    draw.line(scaled_points, fill="#5ad2ff", width=5, joint="curve")
    
    # Car Position
    car_pos = scaled_points[1]
    draw.ellipse(
        (car_pos[0]-8, car_pos[1]-8, car_pos[0]+8, car_pos[1]+8),
        fill="#ff7272", outline="white", width=2
    )
    
    title_font = ensure_font(32, bold=True)
    draw.text((map_x + 20, map_y + 20), "RIM Sport Karting", font=title_font, fill="#e8f6ff")

    # 2. Center Gauge (Speed & RPM)
    gauge_width = 600
    gauge_height = 200
    gauge_x = (width - gauge_width) // 2
    gauge_y = height - gauge_height - 40
    
    # Background
    draw.rounded_rectangle(
        (gauge_x, gauge_y, gauge_x + gauge_width, gauge_y + gauge_height),
        radius=20,
        fill=(10, 14, 20, 200),
        outline="#214d66",
        width=2
    )
    
    # Speed (Large, Center)
    speed_font = ensure_font(120, bold=True)
    unit_font = ensure_font(30, bold=False)
    
    speed_text = "86"
    # Center text logic
    bbox = draw.textbbox((0, 0), speed_text, font=speed_font)
    text_w = bbox[2] - bbox[0]
    text_x = gauge_x + (gauge_width - text_w) // 2
    
    draw.text((text_x, gauge_y + 20), speed_text, font=speed_font, fill="white")
    draw.text((text_x + text_w + 10, gauge_y + 100), "km/h", font=unit_font, fill="#aaaaaa")
    
    # RPM Bar (Bottom of panel)
    bar_x = gauge_x + 40
    bar_y = gauge_y + 140
    bar_w = gauge_width - 80
    bar_h = 30
    
    # Empty bar
    draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), fill=(50, 50, 50, 255))
    
    # Filled bar (80%)
    fill_w = bar_w * 0.8
    draw.rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h), fill="#ff3333") # Redline-ish
    
    # RPM Text
    rpm_font = ensure_font(24, bold=True)
    draw.text((bar_x, bar_y - 30), "10,500 RPM", font=rpm_font, fill="white")
    
    # Gear (Removed as per request)
    # gear_font = ensure_font(80, bold=True)
    # draw.text((gauge_x + 80, gauge_y + 40), "4", font=gear_font, fill="#ffd479")
    # draw.text((gauge_x + 85, gauge_y + 20), "GEAR", font=ensure_font(16), fill="#aaaaaa")

    # 3. Lap List (Bottom Right)
    # Show list of laps, highlight current
    # Columns: Lap | Time | Min/Max RPM | Min/Max Speed
    list_w = 800 # Widened further to prevent overlap
    list_h = 400
    list_x = width - list_w - 60
    list_y = 60 # Moved to top right
    
    draw.rounded_rectangle(
        (list_x, list_y, list_x + list_w, list_y + list_h),
        radius=20,
        fill=(10, 14, 20, 200),
        outline="#214d66",
        width=2
    )
    
    # Header
    header_font = ensure_font(24, bold=True)
    draw.text((list_x + 20, list_y + 20), "LAPS", font=header_font, fill="#e8f6ff")
    
    # Column Headers
    col_font = ensure_font(16, bold=True)
    draw.text((list_x + 250, list_y + 25), "RPM (min/max)", font=col_font, fill="#aaaaaa")
    draw.text((list_x + 500, list_y + 25), "Speed (min/max)", font=col_font, fill="#aaaaaa")
    
    # List items
    item_font = ensure_font(20, bold=False)
    highlight_font = ensure_font(20, bold=True)
    small_font = ensure_font(16, bold=False)
    
    laps = [
        {"id": 1, "time": "01:10.22", "status": "done", "rpm": "4500/12500", "speed": "45/110"},
        {"id": 2, "time": "01:11.05", "status": "done", "rpm": "4600/12400", "speed": "44/108"},
        {"id": 3, "time": "01:09.88", "status": "best", "rpm": "4800/12800", "speed": "48/112"},
        {"id": 4, "time": "01:12.45", "status": "current", "rpm": "4200/11500", "speed": "40/105"},
        {"id": 5, "time": "01:10.50", "status": "future", "rpm": "-/-", "speed": "-/-"},
        {"id": 6, "time": "01:11.20", "status": "future", "rpm": "-/-", "speed": "-/-"},
    ]
    
    start_y = list_y + 60
    row_h = 35
    
    for i, lap in enumerate(laps):
        y = start_y + i * row_h
        
        # Highlight background for current lap
        if lap["status"] == "current":
            draw.rectangle(
                (list_x + 10, y - 5, list_x + list_w - 10, y + 25),
                fill=(90, 210, 255, 50) # Light blue highlight
            )
            
        color = "#aaaaaa"
        font = item_font
        prefix = ""
        
        if lap["status"] == "current":
            color = "#ffffff"
            font = highlight_font
            prefix = "> "
        elif lap["status"] == "best":
            color = "#ffd479" # Gold
            
        # Lap ID
        text = f"{prefix}Lap {lap['id']}"
        draw.text((list_x + 20, y), text, font=font, fill=color)
        
        # Time
        draw.text((list_x + 120, y), lap["time"], font=font, fill=color)
        
        # RPM
        draw.text((list_x + 250, y), lap["rpm"], font=small_font, fill=color)
        
        # Speed
        draw.text((list_x + 500, y), lap["speed"], font=small_font, fill=color)

    # Save
    output_path = Path("render/mockup.png")
    img.save(output_path)
    print(f"Mockup saved to {output_path}")

if __name__ == "__main__":
    draw_mockup(debug=True)
