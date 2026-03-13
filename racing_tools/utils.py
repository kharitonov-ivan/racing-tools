import subprocess

def check_cuda_availability() -> bool:
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

        # Verify NVENC actually works (encode 1 frame with av1_nvenc)
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "lavfi",
             "-i", "nullsrc=s=256x256:d=0.04:r=25",
             "-c:v", "av1_nvenc", "-frames:v", "1",
             "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if probe.returncode != 0:
            return False

        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False