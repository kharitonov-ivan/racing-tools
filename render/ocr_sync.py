
import pytesseract
from PIL import Image
import sys

def main():
    img_path = "render/timestamp_crop.jpg"
    try:
        img = Image.open(img_path)
        
        # Preprocessing
        # Convert to grayscale
        gray = img.convert('L')
        
        # Thresholding (adjust threshold value as needed, e.g. 150-200)
        # Inverting might help if text is white on dark
        # Let's try simple thresholding first
        threshold = 180
        binary = gray.point(lambda p: 255 if p > threshold else 0)
        
        # Save debug image
        binary.save("render/ocr_debug.jpg")
        
        # Use psm 6 (Assume a single uniform block of text) or 7 (Treat the image as a single text line)
        # Add whitelist for digits and time characters if possible (tesseract config)
        custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789:/-APM'
        text = pytesseract.image_to_string(binary, config=custom_config)
        print(f"Raw OCR Result: '{text.strip()}'")
        
        import re
        # Try to find time pattern HH:MM:SS
        time_match = re.search(r'(\d{2}):(\d{2}):?(\d{2})', text)
        if not time_match:
             # Try with missing colons
             time_match = re.search(r'(\d{2}):(\d{2})(\d{2})', text)
             
        if time_match:
            h, m, s = time_match.groups()
            print(f"Detected Time: {h}:{m}:{s}")
        else:
            print("Could not detect time pattern")
            
        # Try to find date pattern
        # Look for 2025
        date_match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](20\d{2})', text)
        if date_match:
            p1, p2, y = date_match.groups()
            print(f"Detected Date: {p1}/{p2}/{y}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
