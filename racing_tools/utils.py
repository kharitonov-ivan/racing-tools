import subprocess

def check_cuda_availability():
    """Check if CUDA hardware acceleration and AV1 NVENC encoder are available."""
    try:
        # Check HW acceleration
        hw_res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        if "cuda" not in hw_res.stdout:
            return False

        # Check Encoder
        enc_res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        if "av1_nvenc" not in enc_res.stdout:
            return False
            
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False