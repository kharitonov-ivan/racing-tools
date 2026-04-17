#!/usr/bin/env python3
"""Main video processing pipeline: telemetry overlay, undistortion, stabilization."""

import argparse
import os
import signal
from pathlib import Path

import ffmpeg
import pandas as pd

from racing_tools.camera.model import CameraModel
from racing_tools.session.crossing_validation import (
    find_crossing_alignment,
    validate_crossings,
)
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
    build_padder,
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
    p.add_argument("--gpx", action="store_true", default=False, help="export session as GPX")

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
            session.detect_sector_crossings()

        if not session.crossings and "Lap" in session.table.columns:
            laps = pd.to_numeric(session.table["Lap"], errors="coerce").ffill().fillna(1)
            t_vals = session.table["Time"].values
            inferred = [float(t_vals[i]) for i in range(1, len(laps)) if laps.iloc[i] != laps.iloc[i - 1]]
            if inferred:
                session.crossings = inferred
                print(f"[Crossings] Inferred {len(inferred)} crossings from Lap column")

        session.add_lap_numbers()

        motec_output = Path(args.telemetry).with_suffix(".ld")
        session.to_motec(output=motec_output, frequency=100.0)
        print(f"[MoTeC] Exported to {motec_output}")

        if args.gpx:
            gpx_output = Path(args.telemetry).with_suffix(".gpx")
            session.to_gpx(gpx_output)
            print(f"[GPX] Exported to {gpx_output}")

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
        existing_times = crossings_sidecar.get("times", []) if crossings_sidecar.exists else []
        times = run_manual_lap_marking(args.inp, start_time=0.0, existing_boundaries=existing_times or None)
        if times:
            crossings_sidecar.save({"times": times})

    crossings_video: list[float] = crossings_sidecar.get("times", [])
    crossings_telem: list[float] = session.crossings if session else []
    sync_mapping: PiecewiseSync | None = None
    video_truncated = False
    crossing_offset = 0

    if session and crossings_video and crossings_telem:
        if len(crossings_video) < len(crossings_telem):
            n_video = len(crossings_video)
            n_telem = len(crossings_telem)
            is_truncated = crossings_sidecar.get("truncated", False)
            if not is_truncated:
                print(
                    f"[Warning] Video has {n_video} crossings but telemetry has {n_telem}. "
                    f"Video may be incomplete (e.g. camera battery died, lost chapter)."
                )
                if args.no_interactive:
                    print("[Error] Cannot resolve crossing mismatch in --no-interactive mode. "
                          "Run once without --no-interactive to mark video as truncated.")
                else:
                    answer = input("Is the video incomplete/corrupt? [y/N]: ").strip().lower()
                    is_truncated = answer == "y"
            if is_truncated:
                crossing_offset = find_crossing_alignment(crossings_video, crossings_telem)
                crossings_telem = crossings_telem[crossing_offset : crossing_offset + n_video]
                video_truncated = True
                if not crossings_sidecar.get("truncated"):
                    crossings_sidecar.save({**crossings_sidecar.data, "truncated": True})
                if crossing_offset == 0:
                    print(f"[Truncated] Video covers laps 1–{n_video - 1} (end missing)")
                elif crossing_offset + n_video == n_telem:
                    print(f"[Truncated] Video covers laps {crossing_offset + 1}–{n_telem - 1} (start missing)")
                else:
                    print(f"[Truncated] Video covers laps {crossing_offset + 1}–{crossing_offset + n_video - 1} (start and end missing)")
            # else: proceed normally, validation will assert mismatch

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

    trim_start = 0.0
    trim_end = video_info.duration
    pad_start = 0.0
    pad_end = 0.0

    if session and sync_mapping is not None:
        telem_times = pd.to_numeric(session.table["Time"], errors="coerce")
        telem_start = float(telem_times.iloc[0])
        telem_end = float(telem_times.iloc[-1])
        telem_start_in_video = float(sync_mapping.telemetry_to_video(telem_start))
        telem_end_in_video = float(sync_mapping.telemetry_to_video(telem_end))
        trim_start = max(0.0, telem_start_in_video)
        trim_end = min(video_info.duration, telem_end_in_video)
        if video_truncated:
            pad_start = max(0.0, trim_start - telem_start_in_video)
            pad_end = max(0.0, telem_end_in_video - trim_end)
            if pad_start > 0:
                print(f"[Pad] Prepending {pad_start:.1f}s of black frames (start missing)")
            if pad_end > 0:
                print(f"[Pad] Appending {pad_end:.1f}s of black frames (end missing)")
        telem_duration = telem_end - telem_start
        video_duration = trim_end - trim_start
        print(f"[Trim] From telemetry range: {trim_start:.1f}s — {trim_end:.1f}s "
              f"(video={video_duration:.1f}s, telem={telem_duration:.1f}s)")
    elif crossings_video:
        TRIM_BUFFER = 5.0
        trim_start = max(0.0, crossings_video[0] - TRIM_BUFFER)
        trim_end = min(video_info.duration, crossings_video[-1] + TRIM_BUFFER)
        print(f"[Trim] Auto from crossings: {trim_start:.1f}s — {trim_end:.1f}s")
    else:
        print(f"[Trim] No crossings, using full video: 0.0s — {trim_end:.1f}s")

    video_session = VideoSession.from_session(session, inp_path)
    video_session._video_info = video_info

    # Capture full telemetry lap stats before resampling replaces the table
    if video_truncated and session:
        video_session.crossings_gps = list(session.crossings)
        video_session._full_lap_stats = video_session.get_lap_stats()
        video_session._full_best_lap = video_session.best_lap
        # Fix "Video LT": only for laps covered by video crossings
        video_durations: dict[int, float] = {}
        for i in range(1, len(crossings_video)):
            video_durations[crossing_offset + i] = crossings_video[i] - crossings_video[i - 1]
        for stat in video_session._full_lap_stats:
            stat["time"] = video_durations.get(stat["id"])

    if session and sync_mapping is not None:
        resample_start = trim_start - pad_start
        resample_end = trim_end + pad_end
        video_session.table = video_session.resample_to_video(
            fps=video_info.fps,
            trim_start=resample_start,
            duration=resample_end - resample_start,
            sync=sync_mapping,
        )

    if video_truncated and session and sync_mapping is not None:
        # Use ALL original telemetry crossings converted to video time
        all_telem_crossings = session.crossings
        all_crossings_in_video = [
            float(sync_mapping.telemetry_to_video(t)) for t in all_telem_crossings
        ]
        video_session.crossings = all_crossings_in_video
        video_session.crossings_gps = list(all_telem_crossings)
        print(f"[Crossings] All telemetry crossings in video time: {len(all_crossings_in_video)} "
              f"(first={all_crossings_in_video[0]:.1f}s, last={all_crossings_in_video[-1]:.1f}s)")
    elif crossings_video:
        video_session.crossings = list(crossings_video)
        print(f"[Crossings] Display crossings (source time): {video_session.crossings[:3]}...")
        if session and sync_mapping is not None:
            video_session.crossings_gps = crossings_telem

    # Transfer sector crossings from telemetry session, converting to video time
    if session and hasattr(session, "sector_crossings") and session.sector_crossings:
        if sync_mapping is not None:
            video_session.sector_crossings = {
                name: [float(sync_mapping.telemetry_to_video(t)) for t in times_list]
                for name, times_list in session.sector_crossings.items()
            }
        else:
            video_session.sector_crossings = dict(session.sector_crossings)

    # Save crossings in original video time for best lap export
    crossings_for_export = list(video_session.crossings)

    # Shift video time and crossings to output time for ASS generation
    ass_time_shift = pad_start - trim_start
    if "VideoTime" in video_session.table.columns:
        video_session.table["VideoTime"] = video_session.table["VideoTime"] + ass_time_shift
    if video_session.crossings:
        video_session.crossings = [c + ass_time_shift for c in video_session.crossings]

    ass = AssBuilder(video_info.width, video_info.height)
    emit_lap_stats_ass(ass, video_session)

    if session:
        emit_gauge_ass(ass, video_session)

    ass_path = ass.write(inp_path.with_suffix(".ass"))
    print(f"[ASS] Exported to {ass_path}")

    if args.no_render:
        return 0

    trimmed_ass_path = Path(args.out).with_name(f"{Path(args.out).stem}_trimmed.ass")
    ass.write_with_offset(trimmed_ass_path, time_offset=0.0)

    pipeline = build_opener(inp_path, hwaccel=hwaccel)
    pipeline = build_trimer(pipeline, trim_start, trim_end)
    if pad_start > 0 or pad_end > 0:
        pipeline = build_padder(pipeline, pad_end=pad_end, pad_start=pad_start)
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
        video_session.crossings = crossings_for_export
        export_best_lap(
            output_video=args.out,
            video_session=video_session,
            video_duration=video_info.duration,
            trim_start=trim_start,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
