import os
from PIL import Image
from tqdm import tqdm

def crop_images(input_dir, output_dir, crop_fraction=0.2):
    """
    Crops the bottom part of images in input_dir and saves them to output_dir,
    preserving EXIF data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(files)} images to process.")

    for filename in tqdm(files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with Image.open(input_path) as img:
                width, height = img.size
                
                # Calculate new height (remove bottom crop_fraction)
                new_height = int(height * (1 - crop_fraction))
                
                # Crop box: (left, top, right, bottom)
                crop_box = (0, 0, width, new_height)
                
                cropped_img = img.crop(crop_box)
                
                # Get EXIF data from original image
                exif_data = img.getexif()
                
                # Save cropped image with original EXIF
                cropped_img.save(output_path, quality=95, exif=exif_data)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    INPUT_DIR = "/mnt/c/Users/supra/Desktop/racing-tools/sfm-sandbox/output_from_video_exif"
    OUTPUT_DIR = "/mnt/c/Users/supra/Desktop/racing-tools/sfm-sandbox/cropped_for_sfm"
    CROP_FRACTION = 0.45 # Removes bottom 45%

    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Crop Fraction: {CROP_FRACTION}")

    crop_images(INPUT_DIR, OUTPUT_DIR, CROP_FRACTION)
    print("Done.")
