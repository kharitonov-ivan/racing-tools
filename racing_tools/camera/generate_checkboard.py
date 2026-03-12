# /// script
# dependencies = [
#   "reportlab",
# ]
# ///

import argparse
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A4
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Generate a checkerboard pattern PDF for camera calibration.')
    parser.add_argument('--rows', type=int, default=6, help='Number of inner corners rows (default: 6). Resulting squares: rows+1')
    parser.add_argument('--cols', type=int, default=9, help='Number of inner corners columns (default: 9). Resulting squares: cols+1')
    parser.add_argument('--size', type=float, default=25.0, help='Size of each square in mm (default: 25)')
    parser.add_argument('--output', default=None, help='Output PDF filename (default: checkerboard_ROWSxCOLS_SIZemm.pdf)')
    
    args = parser.parse_args()

    # Determine output path relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.output is None:
        size_str = f"{int(args.size)}" if args.size.is_integer() else f"{args.size}"
        output_filename = f"checkerboard_{args.rows}x{args.cols}_{size_str}mm.pdf"
        output_path = os.path.join(script_dir, output_filename)
    else:
        output_path = args.output
        # If output is just a filename (no path separators), save to script dir
        if os.path.basename(output_path) == output_path:
            output_path = os.path.join(script_dir, output_path)

    # Calculate actual grid size (squares)
    # Inner corners N means N+1 squares
    grid_rows = args.rows + 1
    grid_cols = args.cols + 1
    
    square_size = args.size * mm
    
    board_width = grid_cols * square_size
    board_height = grid_rows * square_size
    
    # Page settings - use landscape
    from reportlab.lib.pagesizes import landscape
    page_width, page_height = landscape(A4)
    
    if board_width > page_width or board_height > page_height:
        print(f"Warning: Board size ({board_width/mm:.1f}x{board_height/mm:.1f}mm) exceeds page size ({page_width/mm:.1f}x{page_height/mm:.1f}mm).")
    
    c = canvas.Canvas(output_path, pagesize=landscape(A4))
    
    # Calculate starting position to center the board
    start_x = (page_width - board_width) / 2
    start_y = (page_height - board_height) / 2
    
    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    
    print(f"Generating checkerboard: {grid_rows}x{grid_cols} squares ({args.size}mm each)")
    print(f"Inner corners: {args.rows}x{args.cols}")
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Draw black square if (row+col) is odd (or even, depending on preference)
            # Standard chessboards usually have bottom-left black... let's just alternate.
            if (row + col) % 2 == 1:
                x = start_x + col * square_size
                y = start_y + row * square_size
                c.rect(x, y, square_size, square_size, fill=1, stroke=0)
                
    # Add some text/metadata
    c.setFont("Helvetica", 10)
    info_text = f"Checkerboard: {args.rows}x{args.cols} inner corners, {args.size}mm squares"
    c.drawString(start_x, start_y - 15, info_text)
    
    # Add a reference scale (100mm line)
    ref_length_mm = 100.0
    ref_length = ref_length_mm * mm
    ref_y = start_y - 30
    
    # Draw line
    c.line(start_x, ref_y, start_x + ref_length, ref_y)
    # Draw ticks at ends
    c.line(start_x, ref_y - 2*mm, start_x, ref_y + 2*mm)
    c.line(start_x + ref_length, ref_y - 2*mm, start_x + ref_length, ref_y + 2*mm)
    
    c.drawString(start_x + ref_length + 5, ref_y - 3, f"Reference: {ref_length_mm:.0f}mm")
    
    c.save()
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    main()