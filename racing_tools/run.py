#!/usr/bin/env python3
"""Main video processing pipeline: telemetry overlay, undistortion, stabilization."""

import argparse
import os
import signal
from pathlib import Path

import ffmpeg
import pandas as pd

from racing_tools.camera.model import CameraModel
from racing_tools.session.crossing_validation import validate_crossings
from racing_tools.session.distance import ensure_distance
from racing_tools.session.session import (
    PiecewiseSync,
    Session,
    VideoSession,
    create_session_from_crossings,
)
from racing_tools.track.track import Track
from racing_tools.track.visualize_track import plot_track
from racing_tools.utils import check_cuda_availability
from racing_tools.utils.generate_report import generate_report
from racing_tools.utils.sync_ui import run_manual_lap_marking
from racing_tools.video.ass import AssBuilder, emit_gauge_ass, emit_lap_stats_ass
from racing_tools.video.pipeline import (
    Pipeline,
    build_opener,
    build_per_lap_track_maps,
    build_scaler,
    build_stabilizer,
    build_transform_estimator,
    build_trimer,
    build_undistorter,
    build_writer,
    export_best_lap,
)
from racing_tools.video.trim import VideoSidecar
from racing_tools.video.video_info import probe_video


def main() -> int:
    p = argparse.ArgumentParser(description="racing tools: telemetry conversion, report generation, video processing")
    p.add_argument("--in", dest="inp", default=None, help="input video path (optional)")
    p.add_argument("--out", dest="out", help="output video path")
    p.add_argument("--telemetry", dest="telemetry", default=None, help="path to telemetry session")
    p.add_argument("--track", dest="track_dir", default=None, help="path to track directory")
    p.add_argument(
        "--overlay",
        default=None,
        help="overlay image/video path (supports alpha if format has it)",
    )

    p.add_argument("--balance", type=float, default=1.0, help="fisheye balance (0..1)")
    p.add_argument("--fov-scale", type=float, default=1.0, help="fisheye fov_scale (>1 wider)")

    p.add_argument("--stabilise", default=False, action="store_true", help="stabilise video")

    p.add_argument("--shakiness", type=int, default=10, help="vidstabdetect shakiness")
    p.add_argument("--accuracy", type=int, default=15, help="vidstabdetect accuracy")
    p.add_argument("--stepsize", type=int, default=4, help="vidstabdetect stepsize")
    p.add_argument("--smoothing", type=int, default=10, help="vidstabtransform smoothing")
    p.add_argument("--optzoom", type=int, default=0, help="vidstabtransform optzoom")
    p.add_argument("--zoom", type=float, default=0.1, help="vidstabtransform zoom")
    p.add_argument("--crop", default="keep", choices=["black", "keep"], help="Border cropping")
    p.add_argument(
        "--interpol",
        default="bilinear",
        choices=["no", "linear", "bilinear", "bicubic"],
        help="Interpolation",
    )

    p.add_argument("--vcodec", default=None, help="output video codec (default: auto-detect)")
    p.add_argument("--preset", default="7", help="encoder preset (SVT-AV1: 0-13, NVENC: p1-p7)")
    p.add_argument("--crf", type=int, default=28, help="CRF/CQ for AV1 (lower=better, 20-35 recommended)")
    p.add_argument("--resolution", type=int, default=720, help="Output resolution height (e.g. 720)")

    p.add_argument("--intrinsics", help="path to camera intrinsics CSV")
    p.add_argument("--no-interactive", action="store_true", help="skip interactive prompts")

    p.add_argument(
        "--dynamic-overlay",
        action="store_true",
        help="Generate overlay dynamically via pipe",
    )
    p.add_argument(
        "--no-export-best-lap",
        dest="export_best_lap",
        action="store_false",
        default=True,
        help="Disable exporting best lap (enabled by default)",
    )
    p.add_argument("--no-render", action="store_true", help="Generate ASS overlay but skip video render")

    args = p.parse_args()

    session: Session | None = None
    track: Track | None = None
    motec_output: Path | None = None

    if args.track_dir:
        track = Track.load(args.track_dir)
        print(f"[Track] Loaded track from: {args.track_dir}")

    if args.telemetry:
        session = Session.load(args.telemetry)
        if track:
            session.track = track.geometry
            session.detect_crossings()

        if not session.crossings and "Lap" in session.table.columns:
            laps = pd.to_numeric(session.table["Lap"], errors="coerce").ffill().fillna(1)
            t_vals = session.table["Time"].values
            inferred = [float(t_vals[i]) for i in range(1, len(laps)) if laps.iloc[i] != laps.iloc[i - 1]]
            if inferred:
                session.crossings = inferred
                print(f"[Crossings] Inferred {len(inferred)} crossings from Lap column")

        session.add_lap_numbers()

        motec_output = Path(args.telemetry).with_suffix(".ld")
        session.to_motec(output=motec_output, frequency=50.0)
        print(f"[MoTeC] Exported to {motec_output}")

    telemetry_dir = Path(args.telemetry).parent if args.telemetry else Path.cwd()
    report_out = telemetry_dir / f"{telemetry_dir.stem}_report.png"

    if track and args.telemetry:
        plot_track(track=track, track_dir=Path(args.track_dir), output_path=telemetry_dir / "track_visualization.png")
        print(f"[Track] Visualization saved to: {telemetry_dir / 'track_visualization.png'}")
        generate_report(
            telemetry_path=Path(args.telemetry),
            track_dir=Path(args.track_dir),
            output_path=report_out,
        )

    if not args.inp:
        print("[Info] No video input provided. Telemetry processing complete.")
        return 0

    inp_path = Path(args.inp)

    hwaccel = None
    if args.vcodec is None:
        if check_cuda_availability():
            args.vcodec = "av1_nvenc"
            hwaccel = "cuda"
            if args.preset == "7":
                args.preset = "p7"
            print(f"CUDA detected: Using {args.vcodec} with hwaccel={hwaccel}")
        else:
            args.vcodec = "libsvtav1"
            print(f"Using CPU encoder: {args.vcodec}")
    else:
        if "nvenc" in args.vcodec and check_cuda_availability():
            hwaccel = "cuda"

    if not args.out:
        args.out = str(inp_path.with_name(f"{inp_path.stem}_output{inp_path.suffix}"))

    video_info = probe_video(inp_path)

    if args.resolution and video_info.height != args.resolution:
        scale_factor = args.resolution / video_info.height
        new_width = int(video_info.width * scale_factor)
        if new_width % 2 != 0:
            new_width += 1
        print(f"[Scale] Rescaling video from {video_info.width}x{video_info.height} to {new_width}x{args.resolution}")
        video_info.width = new_width
        video_info.height = args.resolution

    crossings_sidecar = VideoSidecar.load(inp_path, "crossings")
    if crossings_sidecar.exists:
        print(f"[Crossings] Found saved video crossings: {len(crossings_sidecar.get('times', []))} laps")
        if not args.no_interactive:
            if input("Regenerate lap markings? [y/N]: ").strip().lower() == "y":
                crossings_sidecar.exists = False

    if not crossings_sidecar.exists and not args.no_interactive:
        times = run_manual_lap_marking(args.inp, start_time=0.0)
        if times:
            crossings_sidecar.save({"times": times})

    crossings_video: list[float] = crossings_sidecar.get("times", [])
    crossings_telem: list[float] = session.crossings if session else []
    sync_mapping: PiecewiseSync | None = None

    if session and crossings_video and crossings_telem:
        validate_crossings(crossings_video, crossings_telem)
        anchors = list(zip(crossings_video, crossings_telem))
        sync_mapping = PiecewiseSync(anchors=anchors)
        print(f"[Sync] Built piecewise mapping with {len(anchors)} anchor points:")
        for i, (v, t) in enumerate(anchors):
            print(f"  Crossing {i + 1}: video={v:.3f}s <-> telem={t:.3f}s (offset={t - v:.3f}s)")

    if not session:
        session = create_session_from_crossings(video_info, crossings_video)
        session.crossings = crossings_video
        session.add_lap_numbers()

    TRIM_BUFFER = 5.0
    trim_start = 0.0
    trim_end = video_info.duration

    if crossings_video:
        trim_start = max(0.0, crossings_video[0] - TRIM_BUFFER)
        trim_end = min(video_info.duration, crossings_video[-1] + TRIM_BUFFER)
        print(f"[Trim] Auto from crossings: {trim_start:.1f}s — {trim_end:.1f}s")
    else:
        print(f"[Trim] No crossings, using full video: 0.0s — {trim_end:.1f}s")

    video_session = VideoSession.from_session(session, inp_path)
    video_session._video_info = video_info

    if session and sync_mapping is not None:
        video_session.table = video_session.resample_to_video(
            fps=video_info.fps,
            trim_start=0.0,
            duration=video_info.duration,
            sync=sync_mapping,
        )

    if crossings_video:
        video_session.crossings = list(crossings_video)
        print(f"[Crossings] Display crossings (source time): {video_session.crossings[:3]}...")

    if session and sync_mapping is not None:
        video_session.crossings_gps = crossings_telem

    ass = AssBuilder(video_info.width, video_info.height)
    emit_lap_stats_ass(ass, video_session)

    if session:
        emit_gauge_ass(ass, video_session)

    ass_path = ass.write(inp_path.with_suffix(".ass"))
    print(f"[ASS] Exported to {ass_path}")

    if args.no_render:
        return 0

    trimmed_ass_path = Path(args.out).with_name(f"{Path(args.out).stem}_trimmed.ass")
    ass.write_with_offset(trimmed_ass_path, time_offset=-trim_start)

    pipeline = build_opener(inp_path, hwaccel=hwaccel)
    pipeline = build_trimer(pipeline, trim_start, trim_end)
    if getattr(args, "resolution", None):
        pipeline = build_scaler(pipeline, video_info.width, video_info.height)
    pipeline = Pipeline(pipeline.video.filter("subtitles", filename=str(trimmed_ass_path)), pipeline.audio)

    if args.intrinsics:
        camera_model = CameraModel.load(Path(args.intrinsics))
        pipeline = build_undistorter(pipeline, camera_model, args.balance, args.fov_scale, video_info)

    if args.stabilise:
        transforms_filepath = inp_path.with_suffix(".trf")

        run_pass1 = True
        if transforms_filepath.exists():
            print(f"Found existing transform file: {transforms_filepath}")
            regenerate = input("Regenerate? [y/N]: ").strip().lower()
            run_pass1 = regenerate == "y"
            if not run_pass1:
                print("Using existing transform file")

        if run_pass1:
            print(f"Pass 1: Detecting stability (shakiness={args.shakiness})...")
            t_pipe = build_transform_estimator(
                pipeline,
                transforms_filepath,
                args.shakiness,
                args.accuracy,
                args.stepsize,
            )
            ffmpeg.output(t_pipe.video, os.devnull, format="null").run(overwrite_output=True)
            print(f"Transform file saved: {transforms_filepath}")

        print(f"Pass 2: Stabilizing (smoothing={args.smoothing})...")
        pipeline = build_stabilizer(
            pipeline,
            transforms_filepath,
            smoothing=args.smoothing,
            zoom=args.zoom,
            optzoom=args.optzoom,
            crop=args.crop,
            interpol=args.interpol,
        )

    if session and track:
        crossings = session.crossings or []
        pipeline = build_per_lap_track_maps(
            pipe=pipeline,
            track=track,
            session_table=video_session.table,
            lap_stats=video_session.get_lap_stats(),
            crossings=crossings,
            width=video_info.width,
            height=video_info.height,
            fps=video_info.fps,
        )

    output = build_writer(pipeline, args.out, vcodec=args.vcodec, preset=args.preset, crf=args.crf)

    original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        output.run()
    except ffmpeg.Error:
        print(f"\n[Interrupted] Output saved to: {args.out}")
        return 0
    finally:
        signal.signal(signal.SIGINT, original_handler)

    if args.export_best_lap:
        export_best_lap(
            output_video=args.out,
            video_session=video_session,
            video_duration=video_info.duration,
            trim_start=trim_start,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
