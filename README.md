# racing-tools

## Docker

```bash
# Build
docker build -t racing-tools .

# Run (CPU)
docker run --rm -v ./data:/app/data racing-tools \
  racing_tools/concat_videos/concat_videos.py data/12-03-2026/

# Run (GPU)
docker run --rm --gpus all -v ./data:/app/data racing-tools \
  racing_tools/concat_videos/concat_videos.py data/12-03-2026/
```