"""FFmpeg pipeline building utilities for video processing."""

import atexit
import os
import tempfile
from pathlib import Path
from typing import NamedTuple

import ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from racing_tools.camera.model import CameraModel
from racing_tools.track.stats import calculate_sector_stats_for_lap
from racing_tools.track.track import Track
from racing_tools.video.ass import ASS_FONT
from racing_tools.video.overlay import draw_track_static
from racing_tools.video.undistort import make_fisheye_remap_maps
from racing_tools.video.video_info import VideoInfo


class Pipeline(NamedTuple):
    """Video and audio stream pair for FFmpeg pipeline."""

    video: ffmpeg.Stream
    audio: ffmpeg.Stream


def build_opener(input_path: str | Path, hwaccel: str | None = None) -> Pipeline:
    v = ffmpeg.input(str(input_path), **({} if hwaccel is None else {"hwaccel": hwaccel}))
    return Pipeline(v, v.audio)


def build_trimer(pipe: Pipeline, ss: float, to: float) -> Pipeline:
    return Pipeline(
        pipe.video.video.filter("trim", start=ss, duration=to - ss).filter("setpts", "PTS-STARTPTS"),
        pipe.audio.filter("atrim", start=ss, duration=to - ss).filter("asetpts", "PTS-STARTPTS"),
    )


def build_scaler(pipe: Pipeline, width: int, height: int) -> Pipeline:
    v = pipe.video.filter("scale", width, height)
    return Pipeline(v, pipe.audio)

def _load_remap_stream(path: Path, fps: float) -> ffmpeg.Stream:
    """Load a PGM remap file as a looped FFmpeg stream."""
    return ffmpeg.input(str(path), loop=1, framerate=fps).video.filter("setpts", "PTS-STARTPTS")


def build_undistorter(pipe: Pipeline, camera_model: CameraModel, balance: float, fov_scale: float, video_info: VideoInfo) -> Pipeline:
    tmp = Path(tempfile.gettempdir())
    xmap, ymap, mask = tmp / "xmap.pgm", tmp / "ymap.pgm", tmp / "mask.pgm"

    make_fisheye_remap_maps(video_info.width, video_info.height, camera_model, xmap, ymap, mask, balance, fov_scale)

    return Pipeline(
        ffmpeg.filter([pipe.video, _load_remap_stream(xmap, video_info.fps), _load_remap_stream(ymap, video_info.fps)], "remap"), pipe.audio
    )


def build_transform_estimator(pipe: Pipeline, transform_path: Path, shakiness: int, accuracy: int, stepsize: int) -> Pipeline:
    v = pipe.video.filter("vidstabdetect", shakiness=shakiness, accuracy=accuracy, stepsize=stepsize, result=str(transform_path))
    return Pipeline(v, pipe.audio)


def build_stabilizer(
    pipe: Pipeline,
    transform_path: Path,
    smoothing: int = 10,
    zoom: int = 0,
    optzoom: int = 0,
    crop: str = "keep",
    interpol: str = "bilinear",
    unsharp: bool = True,
) -> Pipeline:
    v = pipe.video.filter(
        "vidstabtransform", input=str(transform_path), zoom=zoom, smoothing=smoothing, optzoom=optzoom, crop=crop, interpol=interpol
    )
    if unsharp:
        v = v.filter("unsharp", lx=5, ly=5, la=0.8, cx=3, cy=3, ca=0.4)
    return Pipeline(v, pipe.audio)


def build_padder(pipe: Pipeline, pad_end: float = 0.0, pad_start: float = 0.0) -> Pipeline:
    """Pad video with black frames and audio with silence at start/end."""
    v = pipe.video
    a = pipe.audio
    if pad_start > 0:
        v = v.filter("tpad", start_duration=pad_start, color="black")
        a = a.filter("adelay", f"{int(pad_start * 1000)}|{int(pad_start * 1000)}")
    if pad_end > 0:
        v = v.filter("tpad", stop_duration=pad_end, color="black")
        a = a.filter("apad", pad_dur=pad_end)
    return Pipeline(v, a)


def build_ov(pipe: Pipeline, overlay_stream: ffmpeg.Stream | None = None) -> Pipeline:
    v = pipe.video
    if overlay_stream:
        v = v.overlay(overlay_stream, x=0, y=0, eof_action="pass")

    v = v.filter("drawtext", text="%{n}", x=50, y=50, fontsize=48, fontcolor="white", box=1, boxcolor="black@0.5", boxborderw=5)
    return Pipeline(v, pipe.audio)


def build_per_lap_track_maps(
    pipe: Pipeline,
    track: Track,
    session_table,
    lap_stats: list,
    crossings: list,
    width: int,
    height: int,
    fps: float,
) -> Pipeline:
    """
    Build per-lap track map overlays with sector statistics.

    Optimized: 1 static track PNG + ASS subtitles for per-lap stats.
    """
    v = pipe.video

    print(f"[per-lap-map] Starting: track={track is not None}, lap_stats={len(lap_stats) if lap_stats else 0}")

    if not track or not lap_stats:
        print("[per-lap-map] Missing track or lap_stats, skipping")
        return pipe

    sectors = track.segments or []
    print(f"[per-lap-map] Found {len(sectors)} sectors")

    if not sectors:
        print("[per-lap-map] No sectors defined, skipping")
        return pipe

    bounds = track.bounds
    min_x, max_x, min_y, max_y = bounds
    x_range = max_x - min_x or 1
    y_range = max_y - min_y or 1

    scale = height / 1080.0
    map_w, map_h = int(600 * scale), int(600 * scale)
    map_x, map_y = int(30 * scale), int(30 * scale)
    padding = int(20 * scale)

    img = Image.new("RGBA", (map_w, map_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for poly in track.polylines:
        norm_poly = [((p[0] - min_x) / x_range, 1.0 - (p[1] - min_y) / y_range) for p in poly]
        scaled = [(padding + p[0] * (map_w - 2 * padding), padding + p[1] * (map_h - 2 * padding)) for p in norm_poly]
        if len(scaled) > 1:
            draw.line(scaled, fill="#888888", width=max(2, int(4 * scale)))

    sector_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]
    for idx, sector in enumerate(sectors):
        pts = sector.get("points", [])
        if pts:
            start_pt = pts[0]
            norm_x = (start_pt[0] - min_x) / x_range
            norm_y = 1.0 - (start_pt[1] - min_y) / y_range
            sx = padding + norm_x * (map_w - 2 * padding)
            sy = padding + norm_y * (map_h - 2 * padding)

            color = sector_colors[idx % len(sector_colors)]
            r = max(3, int(6 * scale))
            draw.ellipse([(sx - r, sy - r), (sx + r, sy + r)], fill=color, outline="#000000")

    fd, static_track_path = tempfile.mkstemp(suffix="_track.png")
    os.close(fd)
    img.save(static_track_path)
    print(f"[per-lap-map] Saved static track: {static_track_path}")

    sector_positions = []
    for sector in sectors:
        pts = sector.get("points", [])
        if pts:
            mid_idx = len(pts) // 2
            mid_pt = pts[mid_idx]
            norm_x = (mid_pt[0] - min_x) / x_range
            norm_y = 1.0 - (mid_pt[1] - min_y) / y_range
            screen_x = map_x + padding + norm_x * (map_w - 2 * padding)
            screen_y = map_y + padding + norm_y * (map_h - 2 * padding)
            sector_positions.append((screen_x, screen_y))
        else:
            sector_positions.append((0, 0))

    fd_ass, stats_ass_path = tempfile.mkstemp(suffix="_track_stats.ass")
    os.close(fd_ass)

    font_stat = max(10, int(14 * scale))
    font_label = max(14, int(20 * scale))

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TrackStat,{ASS_FONT},{font_stat},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,0,5,0,0,0,1
Style: LapLabel,{ASS_FONT},{font_label},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def fmt_time(t):
        h = int(t / 3600)
        m = int((t % 3600) / 60)
        s = int(t % 60)
        cs = int((t * 100) % 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    with open(stats_ass_path, "w") as f:
        f.write(ass_header)

        for lap in lap_stats:
            lap_id = lap["id"]

            if lap_id == 0:
                continue

            if lap_id - 1 < len(crossings):
                t_start = crossings[lap_id - 1]
            else:
                continue

            if lap_id < len(crossings):
                t_end = crossings[lap_id]
            else:
                t_end = session_table["Time"].max() if "Time" in session_table.columns else t_start + 120

            sector_stats = calculate_sector_stats_for_lap(session_table, lap_id, sectors)

            if not sector_stats:
                continue

            s_str = fmt_time(t_start)
            e_str = fmt_time(t_end)

            for idx, (sx, sy) in enumerate(sector_positions):
                if idx in sector_stats:
                    min_spd, max_spd = sector_stats[idx]
                    txt = f"{int(min_spd)}/{int(max_spd)}"
                    f.write(f"Dialogue: 0,{s_str},{e_str},TrackStat,,0,0,0,,{{\\pos({sx:.0f},{sy:.0f})}}{txt}\n")

            label_x = map_x + map_w // 2
            label_y = map_y + map_h - 10
            f.write(f"Dialogue: 1,{s_str},{e_str},LapLabel,,0,0,0,,{{\\pos({label_x},{label_y})}}Lap {lap_id}\n")

            print(f"[per-lap-map] Lap {lap_id}: {t_start:.1f}s - {t_end:.1f}s, {len(sector_stats)} sectors")

    def cleanup():
        for path in [static_track_path, stats_ass_path]:
            if os.path.exists(path):
                os.unlink(path)

    atexit.register(cleanup)

    track_input = ffmpeg.input(static_track_path, loop=1, framerate=fps)
    v = v.overlay(track_input, x=map_x, y=map_y, eof_action="pass")
    v = v.filter("subtitles", filename=stats_ass_path)

    return Pipeline(v, pipe.audio)


def build_track_map_overlay(
    pipe: Pipeline, track_overlay_data_obj, resampled_df, width, height, fps, lap_stats, seg_stats, best_lap=None
) -> Pipeline:
    """Generates static track map with stats and dynamic ASS events for position."""
    from racing_tools.track.stats import draw_full_track_stats

    v = pipe.video

    if not track_overlay_data_obj:
        return pipe

    track_overlay_data = {
        "segments": track_overlay_data_obj.segments,
        "normalized_lines": track_overlay_data_obj.normalized_lines,
        "start_finish_normalized": track_overlay_data_obj.start_finish_normalized,
        "positions": track_overlay_data_obj.positions,
    }

    best_lap_id = best_lap["id"] if best_lap else -1

    if best_lap_id != -1 and seg_stats:
        track_overlay_data["segment_stats"] = seg_stats
        track_overlay_data["current_lap"] = best_lap_id

    map_w, map_h = 400, 400
    map_x, map_y = 50, 50

    img = Image.new("RGBA", (map_w, map_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    map_box = (0, 0, map_w, map_h)

    drawing_area = draw_track_static(draw, map_box, track_overlay_data)
    draw_full_track_stats(draw, drawing_area, track_overlay_data)

    fd, static_map_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(static_map_path)

    if "NormalizedX" in resampled_df.columns and "NormalizedY" in resampled_df.columns:
        fd_ass, dot_ass_path = tempfile.mkstemp(suffix=".ass")
        os.close(fd_ass)

        ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Dot,{ASS_FONT},60,&H000000FF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        with open(dot_ass_path, "w") as f:
            f.write(ass_header)

            inner_x, inner_y, area_w, area_h = drawing_area
            origin_x = map_x + inner_x
            origin_y = map_y + inner_y

            norm_x = resampled_df["NormalizedX"].fillna(0).values
            norm_y = resampled_df["NormalizedY"].fillna(0).values
            total_frames = len(resampled_df)

            def fmt_time(t):
                h = int(t / 3600)
                m = int((t % 3600) / 60)
                s = int(t % 60)
                cs = int((t * 100) % 100)
                return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

            for i in range(total_frames):
                nx = norm_x[i]
                ny = norm_y[i]

                start_t = i / fps
                end_t = (i + 1) / fps
                s_str = fmt_time(start_t)
                e_str = fmt_time(end_t)

                sx = origin_x + nx * area_w
                sy = origin_y + ny * area_h

                f.write(f"Dialogue: 0,{s_str},{e_str},Dot,,0,0,0,,{{\\pos({sx:.1f},{sy:.1f})}}.\n")

        v = v.filter("subtitles", filename=dot_ass_path)

    v = v.overlay(ffmpeg.input(static_map_path, loop=1, framerate=fps), x=map_x, y=map_y, eof_action="pass")

    return Pipeline(v, pipe.audio)


def build_writer(pipe: Pipeline, output_path, vcodec=None, preset=None, crf=None, bitrate=None):
    """Build FFmpeg output with codec-specific settings."""
    v = pipe.video.filter("format", "yuv420p")

    kwargs = {}
    if vcodec:
        kwargs["vcodec"] = vcodec

    if vcodec and "nvenc" in vcodec:
        if preset:
            kwargs["preset"] = preset
        kwargs["tune"] = "hq"

        if crf is not None:
            kwargs["cq"] = crf
            kwargs["rc"] = "vbr"
            kwargs["b:v"] = "0"
        elif bitrate:
            kwargs["b:v"] = bitrate

        kwargs["g"] = 60
    elif vcodec and "svtav1" in vcodec:
        if preset:
            kwargs["preset"] = preset
        if crf is not None:
            kwargs["crf"] = crf
        if bitrate:
            kwargs["b:v"] = bitrate

        kwargs["g"] = 60
    else:
        if preset:
            kwargs["preset"] = preset
        if crf is not None:
            kwargs["crf"] = crf
        if bitrate:
            kwargs["video_bitrate"] = bitrate

        kwargs["g"] = 60

    return ffmpeg.output(v, pipe.audio, output_path, **kwargs).overwrite_output()


def export_best_lap(
    output_video: str | Path,
    video_session: "VideoSession",
    video_duration: float,
    trim_start: float = 0.0,
    buffer_seconds: float = 3.0,
) -> None:
    """
    Export the best lap from the output video as a separate file.

    Uses video_session.crossings which are already in display time (video - trim).
    """
    best_lap = video_session.best_lap
    if not best_lap:
        print("[Best Lap Export] No valid best lap found.")
        return

    best_lap_id = best_lap["id"]
    print(f"\n[Best Lap Export] Exporting Lap {best_lap_id} ({best_lap['time']:.3f}s)...")

    crossings = getattr(video_session, "crossings", []) or []

    if not crossings or best_lap_id < 1 or best_lap_id > len(crossings):
        print(f"[Best Lap Export] Could not determine crossing times for Lap {best_lap_id}")
        return

    # Crossings are in original video time; convert to output video time
    lap_start = crossings[best_lap_id - 1] - trim_start
    lap_end = (crossings[best_lap_id] if best_lap_id < len(crossings) else crossings[-1] + best_lap["time"]) - trim_start

    clip_start = lap_start - buffer_seconds
    clip_end = lap_end + buffer_seconds

    output_duration = video_duration - trim_start
    clip_start = max(0.0, clip_start)
    clip_end = min(output_duration, clip_end)
    clip_duration = clip_end - clip_start

    print(f"[Best Lap Export] Debug: output clip: start={clip_start:.2f}, end={clip_end:.2f}, duration={clip_duration:.2f}")

    if clip_duration <= 0:
        print("[Best Lap Export] Invalid clip duration, skipping.")
        return

    output_path = Path(output_video)
    best_lap_output = output_path.with_name(f"{output_path.stem}-best-lap{output_path.suffix}")

    print(f"[Best Lap Export] Range: {clip_start:.2f}s to {clip_end:.2f}s (duration: {clip_duration:.2f}s)")

    try:
        (
            ffmpeg.input(str(output_video), ss=clip_start, t=clip_duration)
            .output(str(best_lap_output), vcodec="copy", acodec="copy")
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"[Best Lap Export] Saved to: {best_lap_output}")
    except ffmpeg.Error as e:
        print(f"[Best Lap Export] Failed: {e}")
