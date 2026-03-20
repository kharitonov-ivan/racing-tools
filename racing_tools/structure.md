__init__.py
camera/find_intrinsics.py
├── DEF: main -> [ArgumentParser, 
│   VideoCapture, add_argument, append, 
│   basicConfig, calibrate, 
│   calibrateCamera, copy, cornerSubPix,
│   cvtColor, destroyAllWindows, 
│   dirname, drawChessboardCorners, 
│   error, exists, exit, 
│   findChessboardCorners, imshow, info,
│   isOpened, join, len, parse_args, 
│   range, read, release, reshape, 
│   save_calibration, setNumThreads, 
│   splitext, waitKey, warning, zeros, 
│   zip]
└── DEF: save_calibration -> 
camera/generate_checkboard.py
└── DEF: main -> [ArgumentParser, 
    Canvas, abspath, add_argument, 
    basename, dirname, drawString, int, 
    is_integer, join, landscape, line, 
    parse_args, print, range, rect, 
    save, setFillColorRGB, setFont, 
    setStrokeColorRGB]
camera/model.py
└── CLASS: CameraModel
    ├── DEF: __init__ -> 
    ├── DEF: load -> [FileNotFoundError,
    │   append, array, cls, exists, 
    │   float, get, len, list, open, 
    │   reader]
    ├── DEF: save -> 
    └── DEF: __repr__ -> 
overlay.py
├── CLASS: PredictiveLapModel
│   ├── DEF: __init__ -> 
│   └── DEF: get_time -> 
├── DEF: format_duration -> 
├── DEF: load_session -> 
├── DEF: pick_column
├── DEF: resample_telemetry -> [Index, 
│   ValueError, arange, astype, bfill, 
│   ceil, clip, copy, dropna, 
│   duplicated, ffill, fillna, float, 
│   infer_objects, int, interpolate, 
│   max, min, reindex, reset_index, 
│   round, set_index, sort_index, 
│   sort_values, to_numeric, union, 
│   unique]
├── DEF: ensure_font -> [Path, is_file, 
│   load_default, str, truetype]
├── DEF: normalize_track_polylines -> 
├── DEF: normalize_polyline -> 
├── DEF: normalize_track_positions -> 
├── DEF: detect_crossings -> 
├── DEF: calculate_lap_durations -> 
├── DEF: add_lap_numbers -> 
├── DEF: calculate_lap_stats -> 
├── DEF: select_best_lap -> 
├── CLASS: TrackOverlay
├── DEF: build_track_overlay -> 
│   [TrackOverlay, all, append, bfill, 
│   column_stack, ffill, interpolate, 
│   isna, normalize_polyline, 
│   normalize_track_polylines, 
│   normalize_track_positions, 
│   pick_column, print, to_numeric, 
│   to_numpy, transform]
├── DEF: get_gradient_color -> 
├── DEF: draw_track_static -> 
├── DEF: draw_dynamic_track_map_lines ->
├── DEF: draw_track_stats -> 
├── DEF: draw_track_position -> 
├── DEF: draw_track_map -> 
├── DEF: draw_predictive_delta_static ->
├── DEF: draw_predictive_delta_dynamic 
│   -> 
├── DEF: draw_center_gauge_static -> 
├── DEF: draw_center_gauge_dynamic -> 
├── DEF: draw_static_laptime_table -> 
├── DEF: draw_best_lap_pointer -> 
├── DEF: draw_current_lap_pointer -> 
├── DEF: draw_lap_list -> 
├── DEF: draw_current_lap_counter_static
│   -> 
├── DEF: 
│   draw_current_lap_counter_dynamic -> 
├── DEF: draw_debug_info -> 
├── DEF: generate_static_overlay -> 
│   [Draw, draw_center_gauge_static, 
│   draw_current_lap_counter_static, 
│   draw_predictive_delta_static, 
│   draw_static_laptime_table, 
│   draw_track_static, get, int, len, 
│   load_default, min, new, print, 
│   truetype]
├── DEF: render_info_frame -> [Draw, 
│   alpha_composite, copy, 
│   draw_best_lap_pointer, 
│   draw_center_gauge_dynamic, 
│   draw_current_lap_counter_dynamic, 
│   draw_current_lap_pointer, 
│   draw_debug_info, 
│   draw_dynamic_track_map_lines, 
│   draw_predictive_delta_dynamic, 
│   draw_track_position, 
│   draw_track_stats, enumerate, 
│   format_duration, 
│   generate_static_overlay, get, int, 
│   len, load_default, load_font, max, 
│   min, new, next, print, save, 
│   select_best_lap, sorted, tobytes, 
│   truetype]
├── DEF: get_font_path -> [Path, 
│   is_file, str]
├── DEF: run_ffmpeg -> [Popen, append, 
│   close, enumerate, exit, extend, 
│   flush, get, join, len, lower, print,
│   run, str, wait, write]
├── DEF: parse_args -> [ArgumentParser, 
│   add_argument, parse_args]
├── DEF: has_nvidia_gpu -> 
└── DEF: main -> [DataFrame, 
    FileNotFoundError, Path, 
    PredictiveLapModel, SimpleNamespace,
    SystemExit, TemporaryDirectory, 
    Timestamp, VideoSession, abs, any, 
    append, array, astype, 
    best_lap_generator, bfill, 
    build_track_overlay, 
    calculate_lap_stats, calculate_laps,
    column_stack, copy, cumsum, dropna, 
    enumerate, exists, expanduser, 
    extend, ffill, fillna, float, 
    generate_static_overlay, 
    generate_transforms, get, 
    get_font_path, get_transform_filter,
    has_nvidia_gpu, input, int, 
    interpolate, is_dir, isinstance, 
    isna, isnan, iterrows, len, 
    linspace, load, load_video_metadata,
    loads, lower, max, min, mkdir, norm,
    numeric_s, overlay_generator, 
    parse_args, pick_col, 
    pick_col_local, print, probe_video, 
    project, range, read_text, 
    render_info_frame, replace, 
    resample_telemetry, resolve, 
    run_ffmpeg, run_interactive_sync, 
    select_best_lap, setdefault, sorted,
    str, strip, strptime, sync, 
    timedelta, to_datetime, to_numeric, 
    to_numpy, to_timedelta, tolist, 
    transform, tuple, unique, with_name,
    with_suffix, write_text, zeros, zip]
run.py
├── CLASS: Pipeline
├── DEF: create_session_from_crossings 
│   -> [DataFrame, Session, arange, 
│   array, bisect_right, get_lap_number,
│   sorted]
├── DEF: write_pgm_u16 -> [ValueError, 
│   byteswap, encode, tobytes, 
│   write_bytes]
├── DEF: _compute_undistort_maps -> 
├── DEF: _create_remap_arrays -> 
├── DEF: make_fisheye_remap_maps -> 
│   [_compute_undistort_maps, 
│   _create_remap_arrays, write_pgm_u16]
├── DEF: build_opener -> [Pipeline, 
│   input, str]
├── DEF: build_trimer -> [Pipeline, 
│   filter]
├── DEF: _load_remap_stream -> 
├── DEF: build_undistorter -> [Path, 
│   Pipeline, _load_remap_stream, 
│   filter, gettempdir, 
│   make_fisheye_remap_maps]
├── DEF: build_transform_estimator -> 
│   [Pipeline, filter, str]
├── DEF: build_stabilizer -> [Pipeline, 
│   filter, str]
├── DEF: build_ov -> [Pipeline, filter, 
│   overlay]
├── DEF: build_lap_stats_ov -> 
│   [Pipeline, append, close, enumerate,
│   exists, filter, fmt_ass, 
│   format_duration, get, get_lap_stats,
│   int, isinstance, items, len, list, 
│   mkstemp, next, open, range, 
│   register, replace, sorted, str, sum,
│   title, unlink, write]
├── DEF: build_gauge_overlay -> 
│   [Pipeline, PredictiveLapModel, abs, 
│   bool, ceil, close, copy, exists, 
│   fillna, filter, fmt_lap_time, 
│   fmt_time, get_lap_stats, get_time, 
│   getattr, groupby, int, isnan, len, 
│   list, max, min, mkstemp, open, 
│   print, range, register, to_dict, 
│   unlink, write, zeros, zip]
├── DEF: calculate_segment_stats -> 
├── DEF: draw_full_track_stats -> 
├── DEF: calculate_sector_stats_for_lap 
│   -> 
├── DEF: draw_sectors_with_stats -> 
├── DEF: build_per_lap_track_maps -> 
│   [Draw, Pipeline, append, 
│   calculate_sector_stats_for_lap, 
│   close, ellipse, enumerate, exists, 
│   filter, fmt_time, get, input, int, 
│   len, line, max, mkstemp, new, open, 
│   overlay, print, register, save, 
│   unlink, write]
├── DEF: build_track_map_overlay -> 
│   [Draw, Pipeline, close, 
│   draw_full_track_stats, 
│   draw_track_static, fillna, filter, 
│   fmt_time, input, int, len, mkstemp, 
│   new, open, overlay, range, save, 
│   write]
├── DEF: build_writer -> 
├── DEF: export_best_lap -> [Path, 
│   getattr, input, len, max, min, 
│   output, overwrite_output, print, 
│   run, str, with_name]
└── DEF: main -> [ArgumentParser, Path, 
    PiecewiseSync, add_argument, 
    add_lap_numbers, 
    build_gauge_overlay, 
    build_lap_stats_ov, build_opener, 
    build_per_lap_track_maps, 
    build_stabilizer, 
    build_transform_estimator, 
    build_trimer, build_undistorter, 
    build_writer, 
    check_cuda_availability, 
    create_session_from_crossings, 
    detect_crossings, enumerate, exists,
    export_best_lap, ffill, fillna, 
    float, from_offset, from_session, 
    get, get_lap_stats, get_trim_info, 
    getattr, input, len, list, load, 
    lower, max, min, output, parse_args,
    print, probe_video, range, 
    resample_to_video, run, 
    run_manual_lap_marking, save, 
    signal, str, strip, to_motec, 
    to_numeric, with_name, with_suffix, 
    zip]
session/aim/loader.py
├── DEF: motec_script -> 
│   [FileNotFoundError, is_file]
├── DEF: load_raw -> [AIMXRK, 
│   ChannelNormalizer, 
│   FileNotFoundError, ImportError, 
│   Index, Path, Series, ValueError, 
│   XRK, any, append, array, bfill, 
│   concat, copy, date, duplicated, 
│   ensure_distance, ffill, get, 
│   is_file, isoformat, items, len, 
│   normalize_dataframe, print, 
│   reset_index, resolve, samples, 
│   sort_index, str, strftime, strptime]
└── DEF: load_csv -> [ChannelNormalizer,
    FileNotFoundError, Path, 
    datetime_from_meta, ensure_distance,
    exists, frame, get, 
    infer_datetime_from_path, is_dir, 
    is_file, len, name_tokens, 
    normalize_dataframe, update]
session/aim/utils.py
├── DEF: metadata -> 
├── DEF: datetime_from_meta -> 
└── DEF: frame -> [ValueError, 
    enumerate, isdigit, len, lstrip, 
    metadata, read_csv, read_text, 
    replace, reset_index, splitlines, 
    startswith, str, strip]
session/alfano/loader.py
├── DEF: load_raw -> [ChannelNormalizer,
│   FileNotFoundError, 
│   NotADirectoryError, Path, append, 
│   arange, concat, detect_device, 
│   ensure_distance, enumerate, 
│   extract_lap_number, glob, 
│   infer_datetime_from_path, insert, 
│   is_dir, is_file, len, name_tokens, 
│   normalize_dataframe, read_csv, 
│   sorted]
└── DEF: load_csv -> [ChannelNormalizer,
    FileNotFoundError, Path, 
    ensure_distance, excel_frame, glob, 
    header_clock, 
    infer_datetime_from_path, 
    infer_frequency, is_dir, is_file, 
    len, lower, name_tokens, 
    normalize_dataframe, sorted, 
    stitch_time]
session/alfano/utils.py
├── DEF: extract_lap_number -> 
├── DEF: detect_device -> 
├── DEF: clean_header -> 
├── DEF: header_clock -> 
├── DEF: excel_frame -> 
├── DEF: stitch_time -> [ValueError, 
│   copy, dropna, ffill, fillna, float, 
│   get, map, max, notna, reset_index, 
│   to_numeric, unique]
└── DEF: infer_frequency -> 
session/convert.py
├── DEF: aim_session -> 
├── DEF: alfano_session -> 
├── DEF: alfano_excel_session -> 
├── DEF: run_session -> [Path, 
│   expanduser, print, to_motec]
├── DEF: detect -> 
├── DEF: handle_aim -> [Path, 
│   SystemExit, aim_session, expanduser,
│   is_dir, run_session]
├── DEF: handle_alfano -> [Path, 
│   SystemExit, alfano_session, 
│   expanduser, is_dir, run_session]
├── DEF: handle_alfano_excel -> [Path, 
│   SystemExit, alfano_excel_session, 
│   expanduser, is_dir, run_session]
├── DEF: handle_batch -> [Path, 
│   SystemExit, aim_session, 
│   alfano_excel_session, 
│   alfano_session, detect, expanduser, 
│   is_dir, iterdir, print, rglob, 
│   run_session, sorted, unlink]
├── DEF: handle_mapping -> 
└── DEF: main -> [ArgumentParser, 
    add_argument, add_parser, 
    add_subparsers, func, parse_args, 
    set_defaults]
session/distance.py
├── DEF: distance_from_gps -> 
└── DEF: ensure_distance -> [Series, 
    any, cumsum, diff, 
    distance_from_gps, ffill, fillna, 
    get, len, max, median, next, notna, 
    range, to_numeric]
session/gpx/loader.py
└── DEF: load -> [ChannelNormalizer, 
    DataFrame, FileNotFoundError, 
    ImportError, ValueError, append, 
    arctan2, cos, date, diff, 
    ensure_distance, fillna, 
    fromtimestamp, is_file, isoformat, 
    normalize_dataframe, open, parse, 
    radians, reset_index, shift, sin, 
    sort_values, sqrt, strftime, 
    timestamp]
session/normalizer.py
├── DEF: load_mapping -> 
└── CLASS: ChannelNormalizer
    ├── DEF: __init__ -> [Path, get, 
    │   load_mapping, loads, lower, 
    │   read_text, values]
    ├── DEF: apply_transformations -> 
    └── DEF: normalize_dataframe -> 
session/session.py
├── CLASS: PiecewiseSync
│   ├── DEF: __post_init__ -> 
│   ├── DEF: video_to_telemetry -> 
│   ├── DEF: telemetry_to_video -> 
│   ├── DEF: from_offset -> 
│   ├── DEF: to_dict -> 
│   └── DEF: from_dict -> 
├── CLASS: SessionMetadata
│   └── DEF: copy -> [SessionMetadata, 
│       dict, get]
├── CLASS: Session
│   ├── DEF: driver
│   ├── DEF: driver
│   ├── DEF: venue
│   ├── DEF: venue
│   ├── DEF: vehicle
│   ├── DEF: vehicle
│   ├── DEF: session
│   ├── DEF: session
│   ├── DEF: device
│   ├── DEF: device
│   ├── DEF: event_date
│   ├── DEF: event_date
│   ├── DEF: event_time
│   ├── DEF: event_time
│   ├── DEF: tags
│   ├── DEF: tags
│   ├── DEF: _pick_column
│   ├── DEF: detect_crossings -> 
│   │   [_pick_column, append, fromkeys,
│   │   len, list, range, 
│   │   segments_intersect, to_numeric]
│   ├── DEF: add_lap_numbers -> 
│   ├── DEF: get_lap_durations -> 
│   ├── DEF: _get_gps_lap_durations -> 
│   ├── DEF: get_lap_stats -> 
│   │   [_get_gps_lap_durations, 
│   │   _pick_column, append, get, 
│   │   get_lap_durations, int, max, 
│   │   min, sorted, to_numeric, unique]
│   ├── DEF: best_lap -> 
│   ├── DEF: __repr__ -> 
│   ├── DEF: copy -> [Session, copy]
│   ├── DEF: load -> [FileNotFoundError,
│   │   Path, ValueError, 
│   │   _load_from_zip, glob, is_dir, 
│   │   is_file, list, load_aim_csv, 
│   │   load_aim_raw, load_alfano_csv, 
│   │   load_alfano_raw, load_gpx, 
│   │   lower, startswith]
│   ├── DEF: _load_from_zip -> [Path, 
│   │   ValueError, ZipFile, extractall,
│   │   glob, is_zipfile, list, load, 
│   │   load_alfano_csv, 
│   │   load_alfano_raw, mkdtemp, 
│   │   register, rmtree]
│   ├── DEF: load_aim_raw -> [Path, 
│   │   SessionMetadata, cls, get, 
│   │   load_raw]
│   ├── DEF: load_aim_csv -> [Path, 
│   │   SessionMetadata, cls, get, 
│   │   load_csv]
│   ├── DEF: load_alfano_raw -> [Path, 
│   │   SessionMetadata, cls, get, 
│   │   load_raw]
│   ├── DEF: load_alfano_csv -> [Path, 
│   │   SessionMetadata, cls, get, 
│   │   load_csv]
│   ├── DEF: load_gpx -> [Path, 
│   │   SessionMetadata, cls, get, load]
│   ├── DEF: _ordered_table -> 
│   ├── DEF: to_csv -> [_ordered_table, 
│   │   mkdir, to_csv]
│   ├── DEF: to_motec -> 
│   │   [NamedTemporaryFile, Path, 
│   │   close, extend, getattr, items, 
│   │   mkdir, motec_script, run, str, 
│   │   to_csv, unlink]
│   └── DEF: estimate_laps -> [Series, 
│       append, array, astype, 
│       bisect_right, full, isna, len, 
│       max, next, nonzero, range, 
│       round, segments_intersect, sum, 
│       to_numeric, to_numpy, transform,
│       zip]
└── CLASS: VideoSession
    ├── DEF: __init__ -> [Path, 
    │   SessionMetadata, __init__, 
    │   super]
    ├── DEF: from_session -> 
    ├── DEF: load_with_video -> 
    ├── DEF: info -> [FileNotFoundError,
    │   exists, probe_video]
    ├── DEF: sync -> 
    ├── DEF: save_sync -> 
    ├── DEF: add_video_columns -> 
    └── DEF: resample_to_video -> 
        [DataFrame, arange, copy, int, 
        interp, isinstance, round, 
        sort_values, to_numeric, 
        video_to_telemetry]
session/utils.py
├── DEF: name_tokens -> 
├── DEF: decode_compact_date -> 
├── DEF: decode_time_token -> 
├── DEF: decode_utc_clock -> 
├── DEF: infer_datetime_from_tokens -> 
├── DEF: infer_datetime_from_path -> 
└── DEF: segments_intersect -> 
third_party/MotecLogGenerator/can_utils/
can_utils.py
├── CLASS: CanByteStats
│   ├── DEF: __init__
│   └── DEF: update -> 
├── CLASS: CanFrameStats
│   ├── DEF: __init__ -> 
│   │   [_update_byte_stats, int, len]
│   ├── DEF: update -> 
│   │   [_update_byte_stats, int, len, 
│   │   max, min]
│   ├── DEF: avg_frequency
│   ├── DEF: _update_byte_stats -> 
│   │   [CanByteStats, append, int, len,
│   │   range, str, update]
│   └── DEF: __str__ -> 
├── DEF: parse_can_line -> 
└── DEF: get_id_stats_from_lines -> 
    [CanFrameStats, float, int, len, 
    parse_can_line, update]
third_party/MotecLogGenerator/can_utils/
candump_converter.py
third_party/MotecLogGenerator/can_utils/
dbc_file_from_can_log.py
└── DEF: get_dbc_message_def -> 
third_party/MotecLogGenerator/can_utils/
list_can_ids.py
third_party/MotecLogGenerator/can_utils/
list_can_messages.py
third_party/MotecLogGenerator/data_log.p
y
├── CLASS: DataLog
│   ├── DEF: __init__
│   ├── DEF: clear
│   ├── DEF: add_channel -> [Channel]
│   ├── DEF: start -> 
│   ├── DEF: end -> 
│   ├── DEF: duration -> 
│   ├── DEF: resample -> 
│   ├── DEF: from_can_log -> [Message, 
│   │   __parse_can_log_line, add, 
│   │   add_channel, append, clear, 
│   │   decode_message, 
│   │   get_message_by_frame_id, items, 
│   │   set, zip]
│   ├── DEF: from_csv_log -> [Message, 
│   │   add_channel, append, clear, 
│   │   float, items, len, max, print, 
│   │   split, strip]
│   ├── DEF: from_accessport_log -> 
│   ├── DEF: __parse_can_log_line -> 
│   └── DEF: __str__ -> 
├── CLASS: Channel
│   ├── DEF: __init__ -> 
│   ├── DEF: start
│   ├── DEF: end
│   ├── DEF: avg_frequency -> 
│   ├── DEF: resample -> [Message, 
│   │   append, floor, len, range]
│   └── DEF: __str__ -> 
└── CLASS: Message
    ├── DEF: __init__ -> 
    └── DEF: __str__
third_party/MotecLogGenerator/ldparser/l
dparser.py
├── CLASS: ldData
│   ├── DEF: __init__
│   ├── DEF: __getitem__ -> [Exception, 
│   │   enumerate, isinstance, len]
│   ├── DEF: __iter__ -> 
│   ├── DEF: frompd -> 
│   ├── DEF: fromfile -> 
│   └── DEF: write -> 
├── CLASS: ldEvent
│   ├── DEF: __init__
│   ├── DEF: fromfile -> 
│   ├── DEF: write -> 
│   └── DEF: __str__
├── CLASS: ldVenue
│   ├── DEF: __init__
│   ├── DEF: fromfile -> 
│   ├── DEF: write -> 
│   └── DEF: __str__
├── CLASS: ldVehicle
│   ├── DEF: __init__
│   ├── DEF: fromfile -> 
│   ├── DEF: write -> 
│   └── DEF: __str__
├── CLASS: ldHead
│   ├── DEF: __init__
│   ├── DEF: fromfile -> 
│   ├── DEF: write -> 
│   └── DEF: __str__
├── CLASS: ldChan
│   ├── DEF: __init__
│   ├── DEF: fromfile -> [Exception, 
│   │   calcsize, cls, map, open, read, 
│   │   seek, unpack]
│   ├── DEF: write -> 
│   ├── DEF: data -> [ValueError, 
│   │   fromfile, hex, len, open, pow, 
│   │   print, seek, tell]
│   └── DEF: __str__
├── DEF: decode_string -> 
├── DEF: read_channels -> 
└── DEF: read_ldfile -> 
third_party/MotecLogGenerator/motec_log.
py
└── CLASS: MotecLog
    ├── DEF: __init__ -> 
    ├── DEF: initialize -> 
    ├── DEF: add_channel -> 
    ├── DEF: add_all_channels -> 
    └── DEF: write -> 
third_party/MotecLogGenerator/motec_log_
generator.py
third_party/TestMatLabXRK/test_new_dll.p
y
└── CLASS: TimeStruct
third_party/TrackDataAnalysis/data/__ini
t__.py
third_party/TrackDataAnalysis/data/aim.p
y
└── CLASS: AIM
    ├── DEF: __init__ -> 
    ├── DEF: __del__ -> 
    ├── DEF: close -> 
    ├── DEF: get_vehicle_name -> 
    ├── DEF: get_track_name -> 
    ├── DEF: get_racer_name -> 
    ├── DEF: get_championship_name -> 
    ├── DEF: get_venue_type_name -> 
    ├── DEF: get_laps_count -> 
    ├── DEF: get_lap_info -> 
    ├── DEF: get_channels_count -> 
    ├── DEF: get_channel_name -> 
    ├── DEF: get_channel_units -> 
    ├── DEF: get_channel_samples_count 
    │   -> 
    ├── DEF: get_channel_samples -> 
    ├── DEF: get_GPS_channels_count -> 
    ├── DEF: get_GPS_channel_name -> 
    ├── DEF: get_GPS_channel_units -> 
    ├── DEF: 
    │   get_GPS_channel_samples_count ->
    ├── DEF: get_GPS_channel_samples -> 
    ├── DEF: get_GPS_raw_channels_count 
    │   -> 
    ├── DEF: get_GPS_raw_channel_name ->
    ├── DEF: get_GPS_raw_channel_units 
    │   -> 
    ├── DEF: 
    │   get_GPS_raw_channel_samples_coun
    │   t -> 
    └── DEF: get_GPS_raw_channel_samples
        -> 
third_party/TrackDataAnalysis/data/autos
port_labs.py
├── DEF: _decode_channel_hdr -> 
│   [Channel, array, index, len, list, 
│   reader]
├── DEF: _decode_header -> 
│   [_decode_channel_hdr, list, reader, 
│   zip]
└── DEF: AutosportLabs -> [Lap, LogFile,
    _decode_header, append, array, chr, 
    concatenate, count, enumerate, 
    float, int, localtime, max, open, 
    progress, range, read, readline, 
    rstrip, split, zip]
third_party/TrackDataAnalysis/data/base.
py
├── CLASS: Channel
├── CLASS: Lap
└── CLASS: LogFile
third_party/TrackDataAnalysis/data/dista
nce.py
├── CLASS: ChannelData
│   ├── DEF: from_data -> 
│   ├── DEF: interp -> 
│   ├── DEF: interp_many -> 
│   └── DEF: change_units -> 
└── CLASS: DistanceWrapper
    ├── DEF: __init__ -> 
    │   [_update_time_dist, array]
    ├── DEF: try_gps_lap_insert -> [Lap,
    │   _update_time_dist, array, 
    │   column_stack, enumerate, 
    │   find_laps, get_channel_data, 
    │   get_key_channel_map, len, 
    │   lla2ecef, zip]
    ├── DEF: _calc_time_dist -> 
    ├── DEF: _update_time_dist -> 
    │   [_calc_time_dist]
    ├── DEF: outDist2Time -> 
    ├── DEF: outTime2Dist -> 
    ├── DEF: get_filename
    ├── DEF: get_laps
    ├── DEF: get_metadata
    ├── DEF: get_key_channel_map
    ├── DEF: get_channels -> 
    ├── DEF: get_channel_metadata
    └── DEF: get_channel_data -> 
third_party/TrackDataAnalysis/data/ecuma
ster.py
├── DEF: decode_string_list -> 
├── DEF: decode_groups -> 
├── CLASS: Channel
├── DEF: decode_channels -> [Channel, 
│   append, decode, index, print, range,
│   unpack_from]
├── DEF: expand_repeating_channels -> 
├── DEF: assign_channel_addresses -> 
├── CLASS: AggregateData
│   └── DEF: result -> 
├── CLASS: SecretDecoderRing
│   └── DEF: __init__ -> 
│       [SecretDecoderRing, array, 
│       extend, len, range]
├── DEF: decode_row -> 
│   [SecretDecoderRing, append, array, 
│   extend]
├── DEF: decode_rows -> [AggregateData, 
│   decode_row, len, progress]
├── DEF: assign_data -> 
│   [ThreadPoolExecutor, asarray, 
│   byteswap, cast, column_stack, copy, 
│   list, memoryview, range, result, 
│   submit, values]
├── DEF: decode_len_str -> 
├── DEF: csv_analyze -> 
├── DEF: generate_laps -> [Lap, 
│   column_stack, enumerate, find_laps, 
│   lla2ecef, median, zip]
└── DEF: ECUMASTER_ADU -> [Channel, Lap,
    LogFile, assign_channel_addresses, 
    assign_data, ceil, decode_channels, 
    decode_groups, decode_len_str, 
    decode_rows, decode_string_list, 
    decompress, 
    expand_repeating_channels, 
    generate_laps, int, list, log10, 
    max, open, perf_counter, print, 
    read, unpack_from]
third_party/TrackDataAnalysis/data/gpmf.
py
├── DEF: valid_4cc -> 
├── CLASS: BoxParser
│   ├── DEF: __init__ -> 
│   ├── DEF: count -> 
│   ├── DEF: get_one -> 
│   └── DEF: get_list
├── CLASS: Payload
│   └── DEF: __init__
├── DEF: parse_mp4 -> [BoxParser, 
│   Payload, count, decode, enumerate, 
│   get_list, get_one, len, min, print, 
│   range, unpack, unpack_from, zip]
├── DEF: type_date -> 
├── CLASS: KLV_data
│   ├── DEF: __init__
│   ├── DEF: build_cache -> 
│   ├── DEF: __len__ -> 
│   ├── DEF: __getitem__ -> 
│   └── DEF: __repr__ -> 
├── DEF: KLV_parser -> [KLV_data, 
│   KLV_parser, append, bytes, decode, 
│   len, range, rstrip, unpack_from, 
│   valid_4cc]
└── DEF: MP4_estimate_start_time -> 
    [KLV_parser, abs, append, fileno, 
    len, madvise, memoryview, mmap, 
    open, parse_mp4, sort, sum, 
    timedelta, total_seconds]
third_party/TrackDataAnalysis/data/gps.p
y
├── DEF: llz2web -> 
├── DEF: web2ll -> 
├── DEF: lla2ecef -> 
├── DEF: ecef2lla_osen -> [GPS, abs, 
│   arctan2, cbrt, sqrt, square]
├── DEF: ecef2lla_fukushima2006 -> [GPS,
│   abs, arctan2, copysign, sqrt]
├── DEF: ecef2lla_vermeille2003 -> [GPS,
│   arctan2, cbrt, sqrt]
├── DEF: find_crossing_idx -> 
├── DEF: find_crossing_dist -> 
├── DEF: find_laps -> 
└── DEF: perf_test -> 
third_party/TrackDataAnalysis/data/iraci
ng.py
├── DEF: _dec_str -> 
├── DEF: _decode_var -> [Channel, 
│   _dec_str, copy, ndarray, 
│   unpack_from]
├── DEF: _decode -> [_decode_var, 
│   arange, localtime, print, range, 
│   safe_load, tobytes, unpack_from]
├── DEF: _filter_gps -> 
├── DEF: _find_laps -> [Lap, append, 
│   array, enumerate, max, nonzero, zip]
└── DEF: IRacing -> [LogFile, _decode, 
    _filter_gps, _find_laps, fileno, 
    memoryview, mmap, open]
third_party/TrackDataAnalysis/data/math_
eval.py
├── CLASS: ExprLex
├── CLASS: EvalLiteral
│   ├── DEF: __init__ -> 
│   ├── DEF: timecodes -> 
│   └── DEF: values
├── CLASS: EvalReference
│   ├── DEF: __init__
│   ├── DEF: timecodes -> 
│   └── DEF: values -> 
├── CLASS: EvalOp
│   ├── DEF: __init__ -> 
│   ├── DEF: timecodes -> 
│   └── DEF: values -> [_op, values]
├── CLASS: EvalUser
│   └── DEF: values -> 
├── CLASS: EvalWrap
│   ├── DEF: __init__ -> [__init__, 
│   │   super]
│   └── DEF: values -> 
├── CLASS: ParseError
│   └── DEF: __init__ -> 
├── CLASS: ExprParse
│   ├── DEF: expr -> [EvalReference, _]
│   ├── DEF: expr -> [EvalReference, _]
│   ├── DEF: expr -> [EvalLiteral, _, 
│   │   float]
│   ├── DEF: expr -> [_]
│   ├── DEF: expr -> [EvalOp, _]
│   ├── DEF: expr -> [EvalOp, _]
│   ├── DEF: expr -> [EvalOp, _, keys, 
│   │   len]
│   ├── DEF: expr -> [EvalOp, EvalUser, 
│   │   ParseError, _, append, join, 
│   │   keys, len, sort, str]
│   ├── DEF: expr_list -> [_]
│   ├── DEF: expr_list -> [_]
│   └── DEF: error -> [ParseError, next]
├── DEF: eat_comments
└── DEF: compile -> [EvalWrap, ExprLex, 
    ExprParse, eat_comments, parse, 
    tokenize]
third_party/TrackDataAnalysis/data/megal
og.py
├── DEF: _build_channel -> [Channel, 
│   astype, copy]
├── DEF: _decode -> [Struct, 
│   ThreadPoolExecutor, append, array, 
│   astype, decode, join, keys, len, 
│   localtime, memoryview, ndarray, 
│   print, progress, range, result, 
│   rstrip, submit, sum, unpack_from]
└── DEF: Megalog -> [Lap, LogFile, 
    _decode, fileno, int, max, 
    memoryview, min, mmap, open]
third_party/TrackDataAnalysis/data/motec
.py
├── DEF: _dec_u16 -> 
├── DEF: _dec_u32 -> 
├── DEF: _dec_str -> 
├── DEF: _set_if
├── DEF: _decode_channel -> [Channel, 
│   _dec_str, arange, cast, max, 
│   multiply, unpack_from]
├── DEF: _decode -> [_dec_str, _dec_u16,
│   _dec_u32, _decode_channel, _set_if, 
│   range]
└── DEF: MOTEC -> [Lap, LogFile, 
    _decode, append, enumerate, fileno, 
    int, max, memoryview, mmap, open, 
    values, zip]
third_party/TrackDataAnalysis/data/racel
ogic.py
├── DEF: scan_data -> 
└── DEF: VBOX -> [Channel, Lap, LogFile,
    append, asarray, column_stack, copy,
    enumerate, find, find_laps, float, 
    index, len, lla2ecef, open, 
    readline, rstrip, scan_data, split, 
    startswith, zip]
third_party/TrackDataAnalysis/data/racet
ech.py
├── CLASS: State
├── CLASS: DataView
│   ├── DEF: __init__ -> 
│   └── DEF: ts_lookup -> 
├── DEF: separate_subchannels -> 
├── DEF: to2s16 -> 
├── DEF: new_sector_time -> 
├── DEF: accelerations -> [Channel, 
│   byteswap, to2s16, ts_lookup]
├── DEF: gps_position -> [Channel, 
│   byteswap, ts_lookup]
├── DEF: speed_data -> [Channel, 
│   byteswap, ts_lookup]
├── DEF: rpm -> [Channel, byteswap, 
│   ts_lookup]
├── DEF: analog_input -> [Channel, 
│   byteswap, ts_lookup]
├── DEF: data_storage_channel -> 
├── DEF: external_temperature_sensor -> 
│   [Channel, items, 
│   separate_subchannels, ts_lookup]
├── DEF: external_aux_channel -> 
│   [Channel, items, 
│   separate_subchannels, ts_lookup]
├── DEF: external_angle_channel -> 
│   [Channel, items, 
│   separate_subchannels, ts_lookup]
├── DEF: external_pressure_channel -> 
│   [Channel, items, 
│   separate_subchannels, ts_lookup]
├── DEF: external_miscellaneous_channel 
│   -> [Channel, items, 
│   separate_subchannels, ts_lookup]
├── DEF: ignore
├── DEF: parse -> 
└── DEF: RUN -> [DataView, Lap, LogFile,
    asarray, concatenate, enumerate, 
    extend, max, min, open, parse, 
    perf_counter, print, read, zip]
third_party/TrackDataAnalysis/data/unitc
onv.py
├── CLASS: Unit
├── CLASS: UnitProperty
├── DEF: convert -> 
├── DEF: check_units
├── DEF: display_text -> 
└── DEF: comparable_units -> 
third_party/TrackDataAnalysis/devtools/d
mg.py
third_party/TrackDataAnalysis/devtools/f
ilter_strace.py
third_party/TrackDataAnalysis/devtools/t
est.py
├── DEF: test_aim -> [AIM, 
│   get_GPS_channel_name, 
│   get_GPS_channel_samples, 
│   get_GPS_channel_units, 
│   get_GPS_channels_count, 
│   get_GPS_raw_channel_name, 
│   get_GPS_raw_channel_samples, 
│   get_GPS_raw_channel_units, 
│   get_GPS_raw_channels_count, 
│   get_championship_name, 
│   get_channel_name, get_channel_units,
│   get_channels_count, get_lap_info, 
│   get_laps_count, get_racer_name, 
│   get_track_name, get_vehicle_name, 
│   get_venue_type_name, len, list, 
│   print, range, zip]
├── DEF: test_xrk -> [AIMXRK]
├── DEF: test_xrk_and_ch -> [AIMXRK, 
│   DistanceWrapper, get_channel_data, 
│   get_channels]
├── DEF: ch_help -> [AIMXRK, 
│   _help_decode_channels]
├── DEF: ch_compare -> [AIM, AIMXRK, 
│   abs, all, exit, get_channel_data, 
│   get_channel_name, 
│   get_channel_samples, 
│   get_channel_units, 
│   get_channels_count, int, len, list, 
│   pprint, print, range, sorted, sum, 
│   zip]
└── DEF: gps_ch_compare -> [AIM, AIMXRK,
    get_GPS_channel_name, 
    get_GPS_channel_samples, 
    get_GPS_channels_count, 
    get_channel_data, print, range, zip]
third_party/TrackDataAnalysis/gui.py
├── CLASS: TimeDistStatus
│   ├── DEF: __init__ -> [__init__, 
│   │   connect, super, updateCursor]
│   └── DEF: updateCursor -> 
└── CLASS: MainWindow
    ├── DEF: __init__ -> 
    │   [ChannelsDockWidget, 
    │   ComponentManager, ConfigParser, 
    │   DataDockWidget, DataView, 
    │   LapWidget, LayoutManager, 
    │   MapDockWidget, Maths, QToolBar, 
    │   QVBoxLayout, QWidget, 
    │   TimeDistRef, TimeDistStatus, 
    │   ValuesDockWidget, __init__, 
    │   addAction, addMenu, 
    │   addSeparator, addToolBar, 
    │   addWidget, arguments, connect, 
    │   dumps, exists, fromhex, get, 
    │   load_workspace, loads, makedirs,
    │   menuBar, open_file, read, 
    │   restoreGeometry, restoreState, 
    │   setCentralWidget, setCheckable, 
    │   setContentsMargins, setLayout, 
    │   setMovable, setObjectName, 
    │   setSpacing, set_user_func_dir, 
    │   statusBar, super, update_title, 
    │   writableLocation]
    ├── DEF: sizeHint -> [QSize, 
    │   deviceScale]
    ├── DEF: update_title -> 
    ├── DEF: setup_component_menu -> 
    ├── DEF: show_details -> [QDialog, 
    │   QDialogButtonBox, QTableWidget, 
    │   QTableWidgetItem, QVBoxLayout, 
    │   addWidget, append, basename, 
    │   connect, dirname, exec_, 
    │   get_filename, get_metadata, 
    │   hide, horizontalHeader, items, 
    │   len, setEditTriggers, setItem, 
    │   setLayout, setSectionResizeMode,
    │   setSelectionMode, 
    │   setWindowTitle, sorted, str, 
    │   verticalHeader]
    ├── DEF: toggle_time_dist -> 
    ├── DEF: swap_ref_alt -> 
    ├── DEF: zoom_default -> 
    │   [TimeDistRef, emit]
    ├── DEF: math_editor -> 
    ├── DEF: track_editor -> 
    ├── DEF: toggle_data_offsets -> 
    │   [TimeDistRef, any, emit, 
    │   setChecked, update, warning]
    ├── DEF: preferences -> [QDialog, 
    │   QDialogButtonBox, QFormLayout, 
    │   QGridLayout, QGroupBox, 
    │   QHBoxLayout, QLineEdit, 
    │   QListWidget, QPushButton, 
    │   QToolButton, addItem, addItems, 
    │   addRow, addWidget, connect, 
    │   count, dumps, exec_, get, 
    │   getExistingDirectory, item, 
    │   loads, normpath, range, row, 
    │   selectedIndexes, setIcon, 
    │   setLayout, setText, 
    │   setWindowTitle, 
    │   set_user_func_dir, standardIcon,
    │   style, takeItem, text, 
    │   update_scan_dirs]
    ├── DEF: new_workspace -> 
    ├── DEF: open_workspace -> 
    ├── DEF: load_workspace -> 
    ├── DEF: save_workspace -> 
    ├── DEF: save_as_workspace -> 
    └── DEF: closeEvent -> 
third_party/TrackDataAnalysis/setup.py
third_party/TrackDataAnalysis/sly/__init
__.py
third_party/TrackDataAnalysis/sly/ast.py
└── CLASS: AST
    └── DEF: __init_subclass__ -> 
        [TypeError, getattr, hasattr, 
        isinstance, items, len, list, 
        setattr, zip]
third_party/TrackDataAnalysis/sly/docpar
se.py
└── CLASS: DocParseMeta
    ├── DEF: __new__ -> [__new__, 
    │   isinstance, lexer, parse, 
    │   parser, super, tokenize, update]
    └── DEF: __init_subclass__ -> 
third_party/TrackDataAnalysis/sly/lex.py
├── CLASS: LexError
│   └── DEF: __init__
├── CLASS: PatternError
├── CLASS: LexerBuildError
├── CLASS: LexerStateChange
│   └── DEF: __init__
├── CLASS: Token
│   └── DEF: __repr__
├── CLASS: TokenStr
│   ├── DEF: __new__ -> [__new__, super]
│   ├── DEF: __setitem__
│   └── DEF: __delitem__
├── CLASS: _Before
│   └── DEF: __init__
├── CLASS: LexerMetaDict
│   ├── DEF: __init__
│   ├── DEF: __setitem__ -> 
│   │   [AttributeError, TokenStr, 
│   │   __setitem__, callable, 
│   │   isinstance, super]
│   ├── DEF: __delitem__ -> 
│   │   [__delitem__, append, isupper, 
│   │   super]
│   └── DEF: __getitem__ -> [TokenStr, 
│       __getitem__, isupper, split, 
│       super]
├── CLASS: LexerMeta
│   ├── DEF: __prepare__ -> 
│   │   [LexerMetaDict, hasattr, join]
│   └── DEF: __new__ -> [__new__, 
│       _build, dict, isinstance, items,
│       str, super]
└── CLASS: Lexer
    ├── DEF: _collect_rules -> 
    │   [LexerBuildError, append, 
    │   callable, dict, extend, hasattr,
    │   index, insert, isinstance, 
    │   items, startswith]
    ├── DEF: _build -> [LexerBuildError,
    │   PatternError, _collect_rules, 
    │   add, all, append, callable, 
    │   compile, dict, getattr, 
    │   isinstance, items, join, match, 
    │   set, startswith, update, values,
    │   vars]
    ├── DEF: begin -> [__set_state, 
    │   isinstance]
    ├── DEF: push_state -> 
    ├── DEF: pop_state -> 
    ├── DEF: tokenize -> [Token, 
    │   _set_state, append, end, error, 
    │   get, group, match, pop, type]
    └── DEF: error -> [LexError]
third_party/TrackDataAnalysis/sly/yacc.p
y
├── CLASS: YaccError
├── CLASS: SlyLogger
│   ├── DEF: __init__
│   ├── DEF: debug -> 
│   ├── DEF: warning -> 
│   └── DEF: error -> 
├── CLASS: YaccSymbol
│   ├── DEF: __str__
│   └── DEF: __repr__ -> 
├── CLASS: YaccProduction
│   ├── DEF: __init__
│   ├── DEF: __getitem__
│   ├── DEF: __setitem__
│   ├── DEF: __len__ -> 
│   ├── DEF: lineno -> [AttributeError, 
│   │   getattr]
│   ├── DEF: index -> [AttributeError, 
│   │   getattr]
│   ├── DEF: end -> 
│   ├── DEF: __getattr__ -> 
│   │   [AttributeError, join]
│   └── DEF: __setattr__ -> 
│       [AttributeError, __setattr__, 
│       super]
├── CLASS: Production
│   ├── DEF: __init__ -> 
│   ├── DEF: __str__ -> 
│   ├── DEF: __repr__
│   ├── DEF: __len__ -> 
│   ├── DEF: __nonzero__ -> 
│   │   [RuntimeError]
│   ├── DEF: __getitem__
│   └── DEF: lr_item -> [LRItem, len]
├── CLASS: LRItem
│   ├── DEF: __init__ -> 
│   ├── DEF: __str__ -> 
│   └── DEF: __repr__
├── DEF: rightmost_terminal -> 
├── CLASS: GrammarError
├── CLASS: Grammar
│   ├── DEF: __init__ -> 
│   ├── DEF: __len__ -> 
│   ├── DEF: __getitem__
│   ├── DEF: set_precedence -> 
│   │   [GrammarError]
│   ├── DEF: add_production -> 
│   │   [GrammarError, Production, add, 
│   │   append, enumerate, get, len, 
│   │   rightmost_terminal]
│   ├── DEF: set_start -> [GrammarError,
│   │   Production, append, callable]
│   ├── DEF: find_unreachable -> 
│   ├── DEF: infinite_cycles -> 
│   ├── DEF: undefined_symbols -> 
│   ├── DEF: unused_terminals -> 
│   ├── DEF: unused_rules -> 
│   ├── DEF: unused_precedence -> 
│   ├── DEF: _first -> 
│   ├── DEF: compute_first -> [_first, 
│   │   append]
│   ├── DEF: compute_follow -> [_first, 
│   │   append, compute_first, 
│   │   enumerate, len]
│   ├── DEF: build_lritems -> [LRItem, 
│   │   append, len]
│   └── DEF: __str__ -> 
├── DEF: digraph -> 
├── DEF: traverse -> [FP, R, append, 
│   get, len, min, pop, traverse]
├── CLASS: LALRError
├── CLASS: LRTable
│   ├── DEF: __init__ -> [OrderedDict, 
│   │   build_lritems, compute_first, 
│   │   compute_follow, items, len, 
│   │   list, lr_parse_table, values]
│   ├── DEF: lr0_closure -> 
│   ├── DEF: lr0_goto -> 
│   ├── DEF: lr0_items -> 
│   ├── DEF: 
│   │   compute_nullable_nonterminals ->
│   ├── DEF: 
│   │   find_nonterminal_transitions -> 
│   ├── DEF: dr_relation -> 
│   ├── DEF: reads_relation -> 
│   ├── DEF: compute_lookback_includes 
│   │   -> 
│   ├── DEF: compute_read_sets -> 
│   ├── DEF: compute_follow_sets -> 
│   ├── DEF: add_lookaheads -> 
│   ├── DEF: add_lalr_lookaheads -> 
│   ├── DEF: lr_parse_table -> 
│   │   [LALRError, add_lalr_lookaheads,
│   │   append, enumerate, get, id, 
│   │   join, lr0_goto, lr0_items]
│   └── DEF: __str__ -> 
├── DEF: _collect_grammar_rules -> 
│   [_replace_ebnf_choice, 
│   _replace_ebnf_optional, 
│   _replace_ebnf_repeat, append, 
│   extend, getattr, len, range, split, 
│   unwrap, zip]
├── DEF: _replace_ebnf_repeat -> 
│   [_generate_repeat_rules, 
│   _replace_ebnf_choice, any, index, 
│   list]
├── DEF: _replace_ebnf_optional -> 
│   [_generate_optional_rules, index, 
│   list]
├── DEF: _replace_ebnf_choice -> 
│   [_generate_choice_rules, extend, 
│   len, list, split]
├── DEF: _sanitize_symbols -> 
├── DEF: _generate_repeat_rules -> [_, 
│   _collect_grammar_rules, 
│   _sanitize_symbols, append, extend, 
│   getattr, join, tuple]
├── DEF: _generate_optional_rules -> [_,
│   _collect_grammar_rules, 
│   _sanitize_symbols, extend, join, 
│   len, tuple]
├── DEF: _generate_choice_rules -> [_, 
│   _collect_grammar_rules, 
│   _sanitize_symbols, extend, join]
├── CLASS: ParserMetaDict
│   ├── DEF: __setitem__ -> 
│   │   [GrammarError, __setitem__, 
│   │   callable, hasattr, super]
│   └── DEF: __getitem__ -> 
│       [__getitem__, isupper, super, 
│       upper]
├── DEF: _decorator -> 
├── CLASS: ParserMeta
│   ├── DEF: __prepare__ -> 
│   │   [ParserMetaDict]
│   └── DEF: __new__ -> [__new__, 
│       _build, items, list, super]
└── CLASS: Parser
    ├── DEF: __validate_tokens -> 
    ├── DEF: __validate_precedence -> 
    ├── DEF: __validate_specification ->
    │   [__validate_precedence, 
    │   __validate_tokens]
    ├── DEF: __build_grammar -> 
    │   [Grammar, YaccError, 
    │   _collect_grammar_rules, 
    │   add_production, 
    │   find_unreachable, getattr, 
    │   infinite_cycles, join, len, 
    │   set_precedence, set_start, 
    │   undefined_symbols, 
    │   unused_precedence, unused_rules,
    │   unused_terminals, warning]
    ├── DEF: __build_lrtables -> 
    │   [LRTable, getattr, len, warning]
    ├── DEF: __collect_rules -> 
    ├── DEF: _build -> [YaccError, 
    │   __build_grammar, 
    │   __build_lrtables, 
    │   __collect_rules, 
    │   __validate_specification, get, 
    │   info, open, str, vars, write]
    ├── DEF: error -> 
    ├── DEF: errok
    ├── DEF: restart -> [YaccSymbol, 
    │   append]
    ├── DEF: parse -> [RuntimeError, 
    │   YaccProduction, YaccSymbol, 
    │   append, error, func, get, 
    │   getattr, hasattr, id, len, next,
    │   pop, restart]
    ├── DEF: line_position -> 
    └── DEF: index_position -> 
third_party/TrackDataAnalysis/ui/__init_
_.py
third_party/TrackDataAnalysis/ui/channel
s.py
├── DEF: update_channel_properties -> 
│   [ChannelProperties, get, 
│   get_channel_metadata, get_channels, 
│   items]
├── DEF: channel_color_icon -> [QIcon, 
│   QPainter, QPixmap, fillRect, rect]
├── DEF: add_channel_colors -> [QSize, 
│   addItem, channel_color_icon, len, 
│   range, setIconSize]
├── DEF: channel_editor -> [QComboBox, 
│   QDialog, QDialogButtonBox, 
│   QGridLayout, QLabel, QSpinBox, 
│   addItem, addWidget, 
│   add_channel_colors, adder, 
│   channel_color_icon, 
│   comparable_units, connect, 
│   currentData, emit, exec_, get, 
│   index, rowCount, setCurrentIndex, 
│   setLayout, setMaximum, setMinimum, 
│   setSpecialValueText, setValue, 
│   setWindowTitle, 
│   update_channel_properties, value]
└── DEF: initiate_drag -> [QColor, 
    QDrag, QFont, QFontMetrics, 
    QMimeData, QPainter, QPixmap, QSize,
    begin, deviceScale, drawText, end, 
    exec_, fill, get_channel_prop, 
    height, horizontalAdvance, rect, 
    setFont, setMimeData, setPen, 
    setPixelSize, setPixmap, setText]
third_party/TrackDataAnalysis/ui/compone
nts.py
├── CLASS: ComponentManager
│   ├── DEF: __init__ -> [QAction, 
│   │   __init__, addAction, connect, 
│   │   setContextMenuPolicy, super]
│   ├── DEF: newTDGraph -> 
│   │   [ComponentBase, TimeDist]
│   ├── DEF: newSessionGraph -> 
│   │   [ComponentBase, TimeDist]
│   ├── DEF: newVideo -> [ComponentBase,
│   │   Video]
│   ├── DEF: newTableBuilder -> 
│   │   [ComponentBase, TableBuilder]
│   ├── DEF: paintEvent -> [QColor, 
│   │   QRect, deviceScaleFactor, 
│   │   fillRect, getCoords, int, 
│   │   makePaintHelper, range]
│   ├── DEF: resizeLambda -> [QPointF, 
│   │   height, size, width, x, y]
│   ├── DEF: invertLambda -> [QPointF, 
│   │   height, size, width, x, y]
│   ├── DEF: resizeEvent -> 
│   ├── DEF: updateCursor -> 
│   ├── DEF: updateValues -> 
│   ├── DEF: save_state -> 
│   ├── DEF: load_state -> 
│   │   [ComponentBase, deleteLater, 
│   │   emit, findChildren, setParent]
│   ├── DEF: cut_component -> 
│   ├── DEF: copy_component -> 
│   └── DEF: paste_component -> 
│       [ComponentBase]
├── CLASS: ResizerMode
└── CLASS: ComponentBase
    ├── DEF: __init__ -> [QRectF, 
    │   QVBoxLayout, __init__, 
    │   parentResize, resizeLambda, 
    │   setAttribute, 
    │   setAutoFillBackground, 
    │   setChildWidget, 
    │   setContentsMargins, setFocus, 
    │   setFocusPolicy, 
    │   setMouseTracking, setVisible, 
    │   super]
    ├── DEF: save_state -> 
    ├── DEF: setChildWidget -> [QAction,
    │   addAction, addWidget, connect, 
    │   copy_component, cut_component, 
    │   parent, setMouseTracking, 
    │   setSeparator]
    ├── DEF: parentResize -> [QRectF, 
    │   bottomRight, m, setGeometry, 
    │   toRect, topLeft]
    ├── DEF: saveGeometry -> [QRectF, 
    │   bottomRight, geometry, 
    │   invertLambda, m, parentWidget, 
    │   topLeft]
    ├── DEF: focusInEvent -> 
    ├── DEF: focusOutEvent -> 
    ├── DEF: paintEvent -> [QColor, 
    │   QPen, QPoint, QRectF, adjusted, 
    │   drawRect, fillRect, int, 
    │   makePaintHelper, setPen, 
    │   setWidth]
    ├── DEF: setCursorShape -> [QCursor,
    │   bool, height, setCursor, width, 
    │   x, y]
    ├── DEF: mousePressEvent -> 
    ├── DEF: mouseReleaseEvent -> 
    └── DEF: mouseMoveEvent -> [QPoint, 
        QPointF, QRectF, accept, bottom,
        bottomRight, buttons, geometry, 
        globalPos, height, invertLambda,
        left, m, max, min, 
        minimumSizeHint, mouseMoveEvent,
        move, parentWidget, pos, right, 
        saveGeometry, setBottom, 
        setCursorShape, setGeometry, 
        setLeft, setRight, setTop, 
        super, top, topLeft, width, x, 
        y]
third_party/TrackDataAnalysis/ui/datamgr
.py
├── DEF: closure -> 
├── CLASS: DataModelLap
│   └── DEF: present -> 
├── CLASS: DataModelSection
│   └── DEF: present -> 
├── CLASS: DataDockModel
│   ├── DEF: __init__ -> [QColor, QPen, 
│   │   __init__, super]
│   ├── DEF: set_data -> 
│   └── DEF: present -> 
└── CLASS: DataDockWidget
    ├── DEF: __init__ -> [DataDockModel,
    │   FastItemDelegate, QColor, 
    │   QFileSystemWatcher, QTableView, 
    │   Semaphore, Thread, __init__, 
    │   connect, hide, horizontalHeader,
    │   load_metadata_cache, palette, 
    │   recompute, setColor, 
    │   setContextMenuPolicy, 
    │   setEditTriggers, 
    │   setHighlightSections, 
    │   setHorizontalScrollMode, 
    │   setItemDelegate, 
    │   setMinimumSectionSize, setModel,
    │   setPalette, 
    │   setSectionResizeMode, 
    │   setSelectionBehavior, 
    │   setSelectionMode, setShowGrid, 
    │   setStretchLastSection, 
    │   setWidget, start, statusBar, 
    │   super, update_scan_dirs, 
    │   verticalHeader, 
    │   writableLocation]
    ├── DEF: update_scan_dirs -> 
    ├── DEF: rewrite_metadata_cache -> 
    ├── DEF: load_metadata_cache -> 
    ├── DEF: add_watch_dir -> 
    ├── DEF: stop_metadata_scan -> 
    ├── DEF: process_loop -> 
    ├── DEF: prune_cache -> 
    ├── DEF: process_watch -> 
    │   [DistanceWrapper, LogRef, 
    │   append, builder, close, dumps, 
    │   emit, get_builder, is_dir, len, 
    │   open, pop, prune_cache, 
    │   rewrite_metadata_cache, scandir,
    │   select_track, stat, update_laps,
    │   write]
    ├── DEF: open_from_db -> [QDialog, 
    │   QDialogButtonBox, QGridLayout, 
    │   QLabel, QLineEdit, QListWidget, 
    │   QPushButton, QSize, 
    │   QTableWidget, QTableWidgetItem, 
    │   QVBoxLayout, TextMatcher, 
    │   accept, addItem, addLayout, 
    │   addWidget, all, append, button, 
    │   bytes, clearSelection, closure, 
    │   column, connect, currentItem, 
    │   data, deleteLater, 
    │   devicePointScale, dumps, 
    │   enumerate, exec_, fromhex, get, 
    │   hex, hide, horizontalHeader, 
    │   indexOf, insertWidget, item, 
    │   items, join, keys, len, list, 
    │   loads, match, open_file, range, 
    │   resize, resizeSections, 
    │   restoreGeometry, restoreState, 
    │   saveGeometry, saveState, 
    │   selectedItems, 
    │   setClearButtonEnabled, 
    │   setColumnCount, setData, 
    │   setEditTriggers, 
    │   setHighlightSections, 
    │   setHorizontalHeaderLabels, 
    │   setHorizontalScrollMode, 
    │   setItem, setLayout, 
    │   setPlaceholderText, setRowCount,
    │   setRowHidden, 
    │   setSectionResizeMode, 
    │   setSelected, 
    │   setSelectionBehavior, 
    │   setSelectionMode, setShowGrid, 
    │   setSortingEnabled, setText, 
    │   setWindowTitle, sorted, str, 
    │   takeAt, text, 
    │   update_filter_layout, 
    │   update_matches, values, 
    │   verticalHeader, widget]
    ├── DEF: open_from_file -> 
    ├── DEF: get_builder -> 
    ├── DEF: open_file -> 
    ├── DEF: open_file_worker -> 
    │   [DistanceWrapper, LogRef, 
    │   QProgressDialog, append, 
    │   builder, critical, deleteLater, 
    │   emit, get_builder, len, 
    │   print_exc, reset, select_track, 
    │   setMaximum, setMinimumDuration, 
    │   setValue, setWindowModality, 
    │   try_gps_lap_insert, 
    │   update_channel_properties, 
    │   update_laps, wasCanceled]
    ├── DEF: update_lap_ref
    ├── DEF: close_all_logs -> 
    │   [TimeDistRef, emit]
    ├── DEF: close_one_log -> 
    ├── DEF: context_menu -> [QMenu, 
    │   addAction, basename, 
    │   close_one_log, connect, exec_, 
    │   get_filename, indexAt, 
    │   mapToGlobal, row]
    ├── DEF: clickCell -> [TimeDistRef, 
    │   append, column, emit, len, row]
    └── DEF: recompute -> [DataModelLap,
        DataModelSection, QFont, 
        QFontMetrics, append, 
        clearSpans, devicePointScale, 
        duration, filter, get, 
        get_metadata, height, 
        horizontalAdvance, join, len, 
        range, rowCount, setBold, 
        setColumnWidth, setPixelSize, 
        setRowHeight, setSpan, set_data,
        sorted]
third_party/TrackDataAnalysis/ui/dockers
.py
├── CLASS: FastTableModel
│   ├── DEF: __init__ -> [__init__, 
│   │   super]
│   ├── DEF: set_model_param -> 
│   ├── DEF: headerData
│   ├── DEF: rowCount
│   ├── DEF: columnCount
│   ├── DEF: data
│   └── DEF: present
├── CLASS: FastItemDelegate
│   ├── DEF: __init__ -> [__init__, 
│   │   super]
│   ├── DEF: paint -> 
│   ├── DEF: sizeHint -> [QSize]
│   └── DEF: set_metrics
├── CLASS: TempDockWidget
│   ├── DEF: __init__ -> 
│   │   [RotatedPushButton, __init__, 
│   │   addDockWidget, addWidget, 
│   │   append, connect, hide, 
│   │   setAllowedAreas, setCheckable, 
│   │   setFocusPolicy, setObjectName, 
│   │   super]
│   └── DEF: clicked -> 
├── CLASS: TextMatcher
│   ├── DEF: __init__ -> 
│   └── DEF: match -> 
├── CLASS: ChannelsListWidget
│   ├── DEF: __init__ -> [__init__, 
│   │   super]
│   └── DEF: startDrag -> 
├── CLASS: ChannelsDockWidget
│   ├── DEF: __init__ -> 
│   │   [ChannelsListWidget, QLineEdit, 
│   │   QVBoxLayout, QWidget, 
│   │   TextMatcher, __init__, 
│   │   addWidget, connect, recompute, 
│   │   setClearButtonEnabled, 
│   │   setContextMenuPolicy, 
│   │   setDragDropMode, setDragEnabled,
│   │   setLayout, setPlaceholderText, 
│   │   setWidget, super]
│   ├── DEF: activateItem -> 
│   ├── DEF: textChanged -> 
│   │   [TextMatcher, update_hidden]
│   ├── DEF: context_menu -> [QMenu, 
│   │   addAction, addChannel, 
│   │   addSeparator, channel_editor, 
│   │   channels, connect, exec_, 
│   │   itemAt, mapToGlobal, text]
│   ├── DEF: update_hidden -> 
│   └── DEF: recompute -> [QBrush, 
│       QColor, addItems, channels, 
│       clear, count, currentItem, 
│       findItems, flags, get_channels, 
│       item, range, set, setBackground,
│       setBackgroundColor, 
│       setCurrentItem, setFlags, text, 
│       update_hidden]
├── CLASS: ValuesTableSection
│   ├── DEF: __init__
│   └── DEF: value
├── CLASS: ValuesTableChannel
│   ├── DEF: __init__ -> 
│   ├── DEF: _calc -> 
│   ├── DEF: _format
│   ├── DEF: _format_delta
│   └── DEF: value -> [_calc, _format, 
│       _format_delta]
├── CLASS: ValuesTableFunc
│   ├── DEF: __init__ -> [__init__, 
│   │   super]
│   └── DEF: _calc -> 
├── CLASS: ValuesTableFuncTime
│   ├── DEF: _format -> 
│   └── DEF: _format_delta -> 
├── CLASS: ValuesTableModel
│   ├── DEF: __init__ -> [QColor, QPen, 
│   │   __init__, super]
│   ├── DEF: set_data -> 
│   ├── DEF: update_cursor -> 
│   ├── DEF: present -> 
│   ├── DEF: supportedDragActions
│   └── DEF: flags -> 
├── CLASS: ValuesTableView
│   └── DEF: startDrag -> 
├── CLASS: ValuesDockWidget
│   ├── DEF: __init__ -> 
│   │   [FastItemDelegate, QColor, 
│   │   QLineEdit, QVBoxLayout, QWidget,
│   │   TextMatcher, ValuesTableModel, 
│   │   ValuesTableView, __init__, 
│   │   addWidget, connect, hide, 
│   │   horizontalHeader, palette, 
│   │   recompute, 
│   │   setClearButtonEnabled, setColor,
│   │   setContextMenuPolicy, 
│   │   setDragDropMode, setDragEnabled,
│   │   setEditTriggers, 
│   │   setHighlightSections, 
│   │   setHorizontalScrollMode, 
│   │   setItemDelegate, setLayout, 
│   │   setMinimumSectionSize, setModel,
│   │   setPalette, setPlaceholderText, 
│   │   setSectionResizeMode, 
│   │   setSelectionBehavior, 
│   │   setSelectionMode, setShowGrid, 
│   │   setStretchLastSection, 
│   │   setWidget, super, 
│   │   verticalHeader]
│   ├── DEF: sizeHint -> [QSize]
│   ├── DEF: context_menu -> [QMenu, 
│   │   addAction, addChannel, 
│   │   addSeparator, channel_editor, 
│   │   channels, connect, exec_, 
│   │   indexAt, mapToGlobal, row, type]
│   ├── DEF: activate_cell -> 
│   ├── DEF: section_pressed -> 
│   ├── DEF: text_changed -> 
│   │   [TextMatcher, recompute]
│   ├── DEF: update_cursor -> [QRect, 
│   │   bottomRight, createIndex, 
│   │   topLeft, update, update_cursor, 
│   │   viewport, visualRect]
│   └── DEF: recompute -> [QFont, 
│       QFontMetrics, 
│       ValuesTableChannel, 
│       ValuesTableFunc, 
│       ValuesTableFuncTime, 
│       ValuesTableSection, append, 
│       channels, chr, clearSpans, 
│       columnCount, devicePointScale, 
│       enumerate, height, 
│       horizontalAdvance, keys, len, 
│       list, match, max, ord, range, 
│       rowCount, set, setBold, 
│       setColumnWidth, setPixelSize, 
│       setRowHeight, setRowHidden, 
│       setSpan, set_data, set_metrics, 
│       sort, sorted]
└── CLASS: MapDockWidget
    ├── DEF: __init__ -> [MapWidget, 
    │   __init__, connect, setWidget, 
    │   super]
    └── DEF: update_cursor -> 
third_party/TrackDataAnalysis/ui/layout.
py
├── CLASS: Worksheet
│   ├── DEF: save_state
│   └── DEF: load_state -> [Worksheet]
├── CLASS: Workbook
│   ├── DEF: save_state -> 
│   └── DEF: load_state -> [Workbook, 
│       load_state]
├── CLASS: LayoutTree
│   ├── DEF: setItemDropEnabled -> 
│   ├── DEF: dragEnterEvent -> 
│   └── DEF: dropEvent -> 
├── CLASS: LayoutEditor
│   ├── DEF: __init__ -> [LayoutTree, 
│   │   QDialogButtonBox, QGridLayout, 
│   │   QPushButton, QVBoxLayout, 
│   │   __init__, addLayout, addWidget, 
│   │   connect, fromhex, get, header, 
│   │   hide, insertWorkbook, parent, 
│   │   restoreGeometry, setColumnCount,
│   │   setCurrentItem, 
│   │   setDefaultDropAction, 
│   │   setDragDropMode, 
│   │   setEditTriggers, 
│   │   setItemsExpandable, setLayout, 
│   │   setRowStretch, setWindowTitle, 
│   │   super]
│   ├── DEF: hideEvent -> 
│   ├── DEF: insertWorksheet -> 
│   │   [QTreeWidgetItem, flags, 
│   │   setData, setFlags, setText]
│   ├── DEF: insertWorkbook -> 
│   │   [QTreeWidgetItem, flags, 
│   │   insertWorksheet, parent, 
│   │   setData, setExpanded, setFlags, 
│   │   setText]
│   ├── DEF: selectionChanged -> 
│   ├── DEF: reordered -> 
│   ├── DEF: itemChanged -> 
│   ├── DEF: addWorkbook -> [Workbook, 
│   │   Worksheet, getText, 
│   │   insertWorkbook, parent, 
│   │   reordered, setCurrentItem, 
│   │   topLevelItemCount]
│   ├── DEF: addWorksheet -> [Worksheet,
│   │   childCount, getText, 
│   │   insertWorksheet, len, parent, 
│   │   reordered, selectedItems, 
│   │   setCurrentItem, text]
│   ├── DEF: deleteWorkitem -> 
│   └── DEF: rename -> 
└── CLASS: LayoutManager
    ├── DEF: __init__ -> [QComboBox, 
    │   QHBoxLayout, QTabBar, 
    │   QToolButton, QWidget, __init__, 
    │   addAction, addSeparator, 
    │   addWidget, connect, new_layout, 
    │   setContentsMargins, 
    │   setContextMenuPolicy, setLayout,
    │   setMovable, setSizeAdjustPolicy,
    │   setSpacing, setText, super]
    ├── DEF: tab_context_menu -> [QMenu,
    │   addAction, addSeparator, 
    │   connect, exec_, len, 
    │   mapToGlobal, setCurrentIndex, 
    │   setEnabled, tabAt]
    ├── DEF: new_layout -> [Workbook, 
    │   Worksheet, populateComboBox]
    ├── DEF: layoutEditor -> 
    │   [LayoutEditor, deleteLater, 
    │   exec_, saveCurrentTab]
    ├── DEF: tabMoved -> 
    ├── DEF: tabClicked -> 
    ├── DEF: tabSelected -> 
    ├── DEF: selectSheet -> 
    ├── DEF: populateComboBox -> 
    ├── DEF: workspaceUpdated -> 
    ├── DEF: populateTabBar -> 
    ├── DEF: comboActivated
    ├── DEF: comboChange -> 
    ├── DEF: updateWorkbook -> 
    ├── DEF: rename_worksheet -> 
    ├── DEF: delete_worksheet -> 
    ├── DEF: insert_worksheet_like -> 
    │   [Worksheet, addTab, all, count, 
    │   range, setCurrentIndex, 
    │   setTabData, updateWorkbook]
    ├── DEF: duplicate_worksheet -> 
    ├── DEF: newWorksheet -> [Worksheet,
    │   insert_worksheet_like]
    ├── DEF: newWorkbook -> [Workbook, 
    │   Worksheet, addItem, all, append,
    │   len, range, setCurrentIndex]
    ├── DEF: saveCurrentTab -> 
    ├── DEF: loadCurrentTab -> 
    ├── DEF: save_state -> 
    └── DEF: load_state -> 
third_party/TrackDataAnalysis/ui/map.py
├── DEF: closure -> 
├── DEF: maptiler_get_map -> 
├── CLASS: MapBaseWidget
│   ├── DEF: __init__ -> [QAction, 
│   │   __init__, addAction, connect, 
│   │   setCheckable, setChecked, 
│   │   setContextMenuPolicy, 
│   │   setMinimumSize, super]
│   ├── DEF: handle_update -> [QPixmap, 
│   │   loadFromData, print, result, 
│   │   update]
│   ├── DEF: sizeHint -> [QSize]
│   └── DEF: paint_satellite -> [QColor,
│       QFont, QPen, QPointF, QRectF, 
│       add_done_callback, array, 
│       astype, ceil, closure, 
│       deviceScale, drawPixmap, 
│       drawText, floor, height, int, 
│       isChecked, isinstance, lla2ecef,
│       llz2web, log, max, min, norm, 
│       range, rect, setFont, setPen, 
│       setPixelSize, setStyle, 
│       setWidth, submit, web2ll, width]
└── CLASS: MapWidget
    └── DEF: paintEvent -> [QBrush, 
        QColor, QPen, QPoint, QPointF, 
        QRectF, array, bisect_left, 
        bisect_right, cursor2outTime, 
        deviceScale, drawEllipse, 
        drawLine, fillRect, 
        get_channel_data, 
        get_key_channel_map, get_laps, 
        interp, len, makePaintHelper, 
        max, memoryview, min, 
        paint_satellite, range, 
        setBrush, setPen, setStyle, 
        setWidth]
third_party/TrackDataAnalysis/ui/math.py
├── CLASS: Highlighter
│   ├── DEF: __init__ -> [ExprLex, 
│   │   __init__, super]
│   ├── DEF: maybe_format -> 
│   └── DEF: highlightBlock -> [QColor, 
│       QTextCharFormat, array, compile,
│       cursor2outTime, document, len, 
│       maybe_format, setBackground, 
│       setCurrentBlockState, 
│       setFontItalic, setFontWeight, 
│       setForeground, setText, 
│       toPlainText, tokenize, values]
├── CLASS: ExpressionEditor
│   ├── DEF: __init__ -> [Highlighter, 
│   │   QCheckBox, QComboBox, 
│   │   QDialogButtonBox, QFormLayout, 
│   │   QGridLayout, QLabel, QLineEdit, 
│   │   QPlainTextEdit, __init__, 
│   │   addItem, addLayout, addRow, 
│   │   addWidget, add_channel_colors, 
│   │   connect, document, findData, 
│   │   fromhex, get, restoreGeometry, 
│   │   setChecked, setColumnStretch, 
│   │   setCurrentIndex, setLayout, 
│   │   setWindowTitle, setWordWrap, 
│   │   str, super]
│   ├── DEF: hideEvent -> 
│   └── DEF: validate -> [MathExpr, 
│       accept, compile, currentData, 
│       int, isChecked, text, 
│       toPlainText, warning]
├── CLASS: IndexDetails
├── CLASS: MathTreeModel
│   ├── DEF: __init__ -> [__init__, 
│   │   super]
│   ├── DEF: child -> [IndexDetails, 
│   │   internalPointer, isValid, 
│   │   isinstance, keys, row, sorted]
│   ├── DEF: data -> 
│   ├── DEF: setData -> 
│   ├── DEF: flags -> 
│   ├── DEF: headerData
│   ├── DEF: index -> [QModelIndex, 
│   │   child, createIndex, hasIndex]
│   ├── DEF: parent -> [QModelIndex, 
│   │   createIndex, index, 
│   │   internalPointer, isValid, 
│   │   isinstance, items, sorted]
│   ├── DEF: rowCount -> 
│   ├── DEF: columnCount
│   ├── DEF: supportedDropActions
│   ├── DEF: mimeTypes
│   ├── DEF: mimeData -> [QMimeData, 
│   │   parent, row, setText]
│   ├── DEF: decode_mime -> 
│   ├── DEF: canDropMimeData -> 
│   └── DEF: dropMimeData -> 
├── CLASS: MathEditor
│   ├── DEF: __init__ -> [MathTreeModel,
│   │   QDialogButtonBox, QGridLayout, 
│   │   QPushButton, QTreeView, 
│   │   __init__, addWidget, connect, 
│   │   expandAll, fromhex, get, header,
│   │   restoreGeometry, restoreState, 
│   │   setAcceptDrops, 
│   │   setColumnStretch, 
│   │   setDragDropMode, setDragEnabled,
│   │   setDropIndicatorShown, 
│   │   setLayout, setModel, 
│   │   setWindowTitle, super]
│   ├── DEF: hideEvent -> 
│   ├── DEF: create_group -> [MathGroup,
│   │   createIndex, emit, getText, 
│   │   index, keys, setExpanded, 
│   │   sorted]
│   ├── DEF: get_single_index -> 
│   ├── DEF: get_single_child -> 
│   ├── DEF: comment_something -> 
│   ├── DEF: edit_something -> 
│   ├── DEF: activated_item -> 
│   │   [ExpressionEditor, child, 
│   │   deleteLater, emit, exec_, 
│   │   getText, isinstance, redo_math, 
│   │   warning]
│   ├── DEF: delete_something -> 
│   └── DEF: create_expr -> 
│       [ExpressionEditor, append, 
│       create_group, deleteLater, emit,
│       exec_, getItem, 
│       get_single_child, isinstance, 
│       keys, redo_math, sorted]
├── DEF: math_editor -> [MathEditor, 
│   deleteLater, exec_]
├── DEF: redo_math -> 
├── DEF: channel_editor -> 
│   [ExpressionEditor, channel_editor, 
│   deleteLater, exec_, redo_math]
├── CLASS: MathModuleFinder
│   ├── DEF: __init__ -> [ModuleType, 
│   │   __init__, super]
│   ├── DEF: set_path
│   └── DEF: find_spec -> [ModuleSpec, 
│       SourceFileLoader, join, split, 
│       startswith]
├── DEF: set_user_func_dir -> 
│   [MathModuleFinder, 
│   QFileSystemWatcher, addPath, append,
│   clear_user_module, connect, 
│   directories, removePaths, set_path]
└── DEF: clear_user_module -> 
third_party/TrackDataAnalysis/ui/mpv.py
├── CLASS: ShutdownError
├── CLASS: EventOverflowError
├── CLASS: MpvHandle
├── CLASS: MpvRenderCtxHandle
├── CLASS: PropertyUnavailableError
├── CLASS: ErrorCode
│   ├── DEF: human_readable -> 
│   │   [_mpv_error_string, decode]
│   ├── DEF: default_error_handler -> 
│   │   [ValueError, human_readable]
│   ├── DEF: exception_for_ec -> 
│   └── DEF: raise_for_ec -> 
├── CLASS: MpvOpenGLInitParams
│   └── DEF: __init__
├── CLASS: MpvOpenGLFBO
│   └── DEF: __init__
├── CLASS: MpvRenderFrameInfo
│   └── DEF: as_dict
├── CLASS: MpvOpenGLDRMParams
├── CLASS: MpvOpenGLDRMDrawSurfaceSize
├── CLASS: MpvOpenGLDRMParamsV2
│   └── DEF: __init__
├── CLASS: MpvRenderParam
│   └── DEF: __init__ -> [MpvByteArray, 
│       ValueError, bool, c_char_p, 
│       c_int, c_void_p, cast, cons, 
│       encode, format, int, pointer]
├── DEF: kwargs_to_render_param_array ->
├── CLASS: MpvFormat
│   ├── DEF: __eq__ -> 
│   ├── DEF: __repr__
│   └── DEF: __hash__
├── CLASS: MpvEventID
│   ├── DEF: __repr__ -> 
│   │   [_mpv_event_name, decode]
│   └── DEF: from_str -> 
├── DEF: lazy_decoder -> 
├── CLASS: MpvNodeList
│   ├── DEF: array_value -> 
│   └── DEF: dict_value -> 
├── CLASS: MpvByteArray
│   ├── DEF: __init__ -> 
│   └── DEF: bytes_value -> [POINTER, 
│       cast]
├── CLASS: MpvNode
│   ├── DEF: node_value -> 
│   └── DEF: node_cast_value -> 
│       [TypeError, array_value, bool, 
│       bytes_value, decode, decoder, 
│       dict_value, format, node_value]
├── CLASS: MpvNodeUnion
├── CLASS: MpvEvent
│   ├── DEF: data -> [POINTER, cast, 
│   │   get]
│   ├── DEF: as_dict -> [POINTER, 
│   │   _mpv_event_to_node, 
│   │   _mpv_free_node_contents, cast, 
│   │   create_string_buffer, 
│   │   node_value, pointer, sizeof]
│   └── DEF: __str__ -> 
├── CLASS: MpvEventProperty
│   ├── DEF: name -> 
│   └── DEF: value -> 
├── CLASS: MpvEventLogMessage
│   ├── DEF: prefix -> 
│   ├── DEF: level -> 
│   └── DEF: text -> 
├── CLASS: MpvEventEndFile
├── CLASS: MpvEventStartFile
├── CLASS: MpvEventClientMessage
│   └── DEF: args -> 
├── CLASS: MpvEventCommand
│   ├── DEF: unpack -> 
│   └── DEF: result -> 
├── CLASS: MpvEventHook
│   └── DEF: name -> 
├── CLASS: StreamCallbackInfo
├── DEF: _handle_func -> 
├── DEF: bytes_free_errcheck -> 
│   [_mpv_free, cast, notnull_errcheck]
├── DEF: notnull_errcheck -> 
│   [RuntimeError, format]
├── DEF: _mpv_client_api_version -> 
├── DEF: _mpv_coax_proptype -> 
│   [TypeError, encode, format, 
│   proptype, str, type]
├── DEF: _make_node_str_list -> 
│   [MpvNode, MpvNodeList, MpvNodeUnion,
│   _mpv_coax_proptype, c_char_p, cast, 
│   len, pointer]
├── DEF: _make_node_str_map -> [MpvNode,
│   MpvNodeList, MpvNodeUnion, 
│   _mpv_coax_proptype, c_char_p, cast, 
│   encode, items, len, pointer]
├── DEF: _event_generator -> 
│   [StopIteration, _mpv_wait_event]
├── DEF: _create_null_term_cmd_arg_array
│   -> 
├── CLASS: _Proxy
│   └── DEF: __init__ -> [__setattr__, 
│       super]
├── CLASS: _PropertyProxy
│   └── DEF: __dir__ -> [__dir__, 
│       replace, super]
├── CLASS: _FileLocalProxy
│   ├── DEF: __getitem__ -> 
│   │   [__getitem__]
│   ├── DEF: __setitem__ -> 
│   │   [__setitem__]
│   └── DEF: __iter__ -> 
├── CLASS: _OSDPropertyProxy
│   ├── DEF: __getattr__ -> 
│   │   [_get_property, _py_to_mpv]
│   └── DEF: __setattr__ -> 
│       [AttributeError]
├── CLASS: _DecoderPropertyProxy
│   ├── DEF: __init__ -> [__init__, 
│   │   __setattr__, super]
│   ├── DEF: __getattr__ -> 
│   │   [_get_property, _py_to_mpv]
│   └── DEF: __setattr__ -> [_py_to_mpv,
│       setattr]
├── CLASS: GeneratorStream
│   ├── DEF: __init__
│   ├── DEF: seek -> [_generator_fun, 
│   │   iter]
│   ├── DEF: read -> 
│   ├── DEF: close -> 
│   └── DEF: cancel -> 
├── CLASS: ImageOverlay
│   ├── DEF: __init__ -> 
│   ├── DEF: update -> 
│   └── DEF: remove -> 
├── CLASS: FileOverlay
│   ├── DEF: __init__ -> 
│   ├── DEF: update -> 
│   └── DEF: remove -> 
├── CLASS: MPV
│   ├── DEF: __init__ -> [Lock, Thread, 
│   │   _DecoderPropertyProxy, 
│   │   _FileLocalProxy, 
│   │   _OSDPropertyProxy, _mpv_create, 
│   │   _mpv_create_client, 
│   │   _mpv_initialize, 
│   │   _mpv_set_option_string, 
│   │   defaultdict, encode, istr, 
│   │   items, register_stream_protocol,
│   │   replace, set, set_loglevel, 
│   │   start, str, type]
│   ├── DEF: _enqueue_exceptions -> 
│   ├── DEF: _loop -> 
│   │   [EventOverflowError, 
│   │   ShutdownError, 
│   │   _enqueue_exceptions, 
│   │   _event_generator, _log_handler, 
│   │   _mpv_destroy, callback, cb, 
│   │   decode, exception_for_ec, 
│   │   format_exc, handler, list, pop, 
│   │   values, warn]
│   ├── DEF: core_shutdown
│   ├── DEF: check_core_alive -> 
│   │   [ShutdownError]
│   ├── DEF: wait_until_paused -> 
│   ├── DEF: wait_for_playback -> 
│   ├── DEF: wait_until_playing -> 
│   ├── DEF: wait_for_property -> 
│   ├── DEF: wait_for_shutdown -> 
│   ├── DEF: _set_error_handler -> 
│   │   [EventOverflowError, 
│   │   ShutdownError, event_callback, 
│   │   set_exception]
│   ├── DEF: 
│   │   prepare_and_wait_for_property ->
│   │   [Future, _set_error_handler, 
│   │   add, check_core_alive, cond, 
│   │   discard, err_unregister, 
│   │   getattr, observe_property, 
│   │   replace, result, set_exception, 
│   │   set_result, 
│   │   set_running_or_notify_cancel, 
│   │   unobserve_property]
│   ├── DEF: wait_for_event -> 
│   ├── DEF: prepare_and_wait_for_event 
│   │   -> [Future, _set_error_handler, 
│   │   add, check_core_alive, cond, 
│   │   discard, err_unregister, 
│   │   event_callback, result, 
│   │   set_exception, set_result, 
│   │   set_running_or_notify_cancel, 
│   │   unregister_mpv_events]
│   ├── DEF: __del__ -> 
│   ├── DEF: terminate -> [UserWarning, 
│   │   _mpv_terminate_destroy, 
│   │   current_thread, join, quit]
│   ├── DEF: set_loglevel -> 
│   │   [_mpv_request_log_messages, 
│   │   encode]
│   ├── DEF: string_command -> 
│   │   [_create_null_term_cmd_arg_array
│   │   , _mpv_command]
│   ├── DEF: command_async -> [Future, 
│   │   POINTER, ValueError, 
│   │   _make_node_str_list, 
│   │   _make_node_str_map, 
│   │   _mpv_abort_async_command, 
│   │   _mpv_command_node_async, 
│   │   callback, cast, id, 
│   │   set_exception, set_result, 
│   │   set_running_or_notify_cancel, 
│   │   unpack]
│   ├── DEF: node_command -> 
│   ├── DEF: command -> [POINTER, 
│   │   ValueError, _make_node_str_list,
│   │   _make_node_str_map, 
│   │   _mpv_command_node, 
│   │   _mpv_free_node_contents, cast, 
│   │   create_string_buffer, 
│   │   node_value, sizeof]
│   ├── DEF: seek -> 
│   ├── DEF: revert_seek -> 
│   ├── DEF: frame_step -> 
│   ├── DEF: frame_back_step -> 
│   ├── DEF: property_add -> 
│   ├── DEF: property_multiply -> 
│   ├── DEF: cycle -> 
│   ├── DEF: screenshot -> 
│   ├── DEF: screenshot_to_file -> 
│   ├── DEF: screenshot_raw -> 
│   │   [ValueError, command, format, 
│   │   frombytes, merge, split]
│   ├── DEF: allocate_overlay_id -> 
│   │   [IndexError, add, range, set, 
│   │   sorted]
│   ├── DEF: free_overlay_id -> 
│   ├── DEF: create_file_overlay -> 
│   │   [FileOverlay, 
│   │   allocate_overlay_id]
│   ├── DEF: create_image_overlay -> 
│   │   [ImageOverlay, 
│   │   allocate_overlay_id]
│   ├── DEF: remove_overlay -> 
│   ├── DEF: playlist_next -> 
│   ├── DEF: playlist_prev -> 
│   ├── DEF: playlist_play_index -> 
│   ├── DEF: _encode_options -> 
│   │   [_py_to_mpv, format, items, 
│   │   join, str]
│   ├── DEF: loadfile -> 
│   │   [_encode_options, command, 
│   │   encode]
│   ├── DEF: loadlist -> 
│   ├── DEF: playlist_clear -> 
│   ├── DEF: playlist_remove -> 
│   ├── DEF: playlist_move -> 
│   ├── DEF: playlist_shuffle -> 
│   ├── DEF: playlist_unshuffle -> 
│   ├── DEF: run -> 
│   ├── DEF: quit -> 
│   ├── DEF: quit_watch_later -> 
│   ├── DEF: stop -> 
│   ├── DEF: audio_add -> [_drop_nones, 
│   │   command, encode]
│   ├── DEF: audio_remove -> 
│   ├── DEF: audio_reload -> 
│   ├── DEF: video_add -> [_drop_nones, 
│   │   command, encode]
│   ├── DEF: video_remove -> 
│   ├── DEF: video_reload -> 
│   ├── DEF: sub_add -> [_drop_nones, 
│   │   command, encode]
│   ├── DEF: sub_remove -> 
│   ├── DEF: sub_reload -> 
│   ├── DEF: sub_step -> 
│   ├── DEF: sub_seek -> 
│   ├── DEF: toggle_osd -> 
│   ├── DEF: print_text -> 
│   ├── DEF: show_text -> 
│   ├── DEF: expand_text -> 
│   ├── DEF: expand_path -> 
│   ├── DEF: show_progress -> 
│   ├── DEF: rescan_external_files -> 
│   ├── DEF: discnav -> 
│   ├── DEF: mouse -> 
│   ├── DEF: keypress -> 
│   ├── DEF: keydown -> 
│   ├── DEF: keyup -> 
│   ├── DEF: keybind -> 
│   ├── DEF: write_watch_later_config ->
│   ├── DEF: overlay_add -> 
│   ├── DEF: overlay_remove -> 
│   ├── DEF: osd_overlay -> 
│   ├── DEF: osd_overlay_remove -> 
│   ├── DEF: script_message -> 
│   ├── DEF: script_message_to -> 
│   ├── DEF: drop_buffers -> 
│   ├── DEF: vf_command -> 
│   ├── DEF: af_command -> 
│   ├── DEF: observe_property -> 
│   │   [_mpv_observe_property, append, 
│   │   encode, hash]
│   ├── DEF: property_observer -> 
│   ├── DEF: unobserve_property -> 
│   │   [_mpv_unobserve_property, hash, 
│   │   remove]
│   ├── DEF: unobserve_all_properties ->
│   ├── DEF: register_message_handler ->
│   │   [_register_message_handler_inter
│   │   nal]
│   ├── DEF: 
│   │   _register_message_handler_intern
│   │   al
│   ├── DEF: unregister_message_handler 
│   │   -> 
│   ├── DEF: message_handler -> 
│   │   [_register_message_handler_inter
│   │   nal, unregister_message_handler]
│   ├── DEF: register_event_callback -> 
│   ├── DEF: unregister_event_callback 
│   │   -> 
│   ├── DEF: event_callback -> 
│   ├── DEF: _binding_name -> 
│   ├── DEF: on_key_press -> 
│   ├── DEF: key_binding -> 
│   ├── DEF: register_key_binding -> 
│   │   [TypeError, ValueError, 
│   │   _binding_name, callable, 
│   │   command, format, isinstance, 
│   │   match, register_message_handler]
│   ├── DEF: _handle_key_binding_message
│   │   -> 
│   ├── DEF: unregister_key_binding -> 
│   │   [_binding_name, command, 
│   │   unregister_message_handler]
│   ├── DEF: register_stream_protocol ->
│   │   [KeyError, StreamCancelFn, 
│   │   StreamCloseFn, StreamReadFn, 
│   │   StreamSeekFn, StreamSizeFn, 
│   │   _enqueue_exceptions, 
│   │   _mpv_stream_cb_add_ro, c_void_p,
│   │   cancel, close, decode, 
│   │   decorator, encode, format_exc, 
│   │   hasattr, len, open_fn, range, 
│   │   read, seek, set_exception, warn]
│   ├── DEF: play -> 
│   ├── DEF: playlist_filenames
│   ├── DEF: playlist_append -> 
│   ├── DEF: _python_stream_open -> 
│   │   [GeneratorStream, ValueError, 
│   │   _python_stream_catchall, 
│   │   fullmatch, groups]
│   ├── DEF: python_stream -> [KeyError,
│   │   RuntimeError, format]
│   ├── DEF: play_context -> [Queue, 
│   │   _getframe, get, hash, play, put,
│   │   python_stream, unregister]
│   ├── DEF: play_bytes -> [_getframe, 
│   │   hash, play, python_stream, 
│   │   unregister]
│   ├── DEF: python_stream_catchall -> 
│   │   [KeyError, RuntimeError]
│   ├── DEF: _get_property -> [POINTER, 
│   │   TypeError, 
│   │   _mpv_free_node_contents, 
│   │   _mpv_get_property, cast, 
│   │   check_core_alive, 
│   │   create_string_buffer, decode, 
│   │   encode, node_value, sizeof]
│   ├── DEF: _set_property -> 
│   │   [_make_node_str_list, 
│   │   _mpv_coax_proptype, 
│   │   _mpv_set_property, 
│   │   _mpv_set_property_string, 
│   │   check_core_alive, encode, 
│   │   isinstance]
│   ├── DEF: __getattr__ -> 
│   │   [_get_property, _py_to_mpv]
│   ├── DEF: __setattr__ -> 
│   │   [__setattr__, _py_to_mpv, 
│   │   _set_property, startswith, 
│   │   super]
│   ├── DEF: __dir__ -> [__dir__, 
│   │   replace, super]
│   ├── DEF: properties -> 
│   ├── DEF: __getitem__ -> 
│   │   [_get_property]
│   ├── DEF: __setitem__ -> 
│   │   [_set_property]
│   ├── DEF: __iter__ -> 
│   └── DEF: option_info -> 
│       [_get_property]
└── CLASS: MpvRenderContext
    ├── DEF: __init__ -> [POINTER, 
    │   _mpv_render_context_create, 
    │   cast, create_string_buffer, 
    │   kwargs_to_render_param_array, 
    │   sizeof]
    ├── DEF: free -> 
    │   [_mpv_render_context_free]
    ├── DEF: __setattr__ -> 
    │   [MpvRenderParam, RenderUpdateFn,
    │   __setattr__, 
    │   _mpv_render_context_set_paramete
    │   r, 
    │   _mpv_render_context_set_update_c
    │   allback, func, startswith, 
    │   super]
    ├── DEF: __getattr__ -> 
    │   [MpvRenderParam, POINTER, 
    │   _mpv_render_context_get_info, 
    │   as_dict, cast, 
    │   create_string_buffer, sizeof, 
    │   type]
    ├── DEF: update -> 
    │   [_mpv_render_context_update, 
    │   bool]
    ├── DEF: render -> 
    │   [_mpv_render_context_render, 
    │   kwargs_to_render_param_array]
    └── DEF: report_swap -> 
        [_mpv_render_context_report_swap
        ]
third_party/TrackDataAnalysis/ui/state.p
y
├── CLASS: ChannelProperties
├── CLASS: ChannelData
│   └── DEF: derive -> 
├── CLASS: MathExpr
├── CLASS: MathGroup
├── CLASS: Maths
│   ├── DEF: update_channel_map -> 
│   └── DEF: get_channel_data -> 
│       [ChannelData, add, append, 
│       arange, int, isfinite, len, 
│       linspace, list, max, min, 
│       outTime2Dist, pop, set, 
│       timecodes, values]
├── CLASS: TimeDistRef
├── CLASS: LogRef
│   ├── DEF: get_channel_data -> 
│   ├── DEF: update_laps -> [LapRef, 
│   │   TimeDistRef, duration, get_laps,
│   │   len, median, min, outTime2Dist]
│   └── DEF: math_invalidate
├── CLASS: LapRef
│   ├── DEF: lapDist2Time -> 
│   ├── DEF: lapTime2Dist -> 
│   ├── DEF: offDist2Time -> 
│   ├── DEF: duration
│   └── DEF: get_channel_data -> 
├── CLASS: Marker
├── CLASS: Sectors
├── CLASS: Track
│   └── DEF: __init__ -> 
├── CLASS: DataView
│   ├── DEF: get_laps -> 
│   ├── DEF: outTime2Mode -> 
│   ├── DEF: outMode2Dist -> 
│   ├── DEF: outMode2Time -> 
│   ├── DEF: lapTime2Mode -> 
│   ├── DEF: lapMode2Dist -> 
│   ├── DEF: lapDist2Mode -> 
│   ├── DEF: offTime2outMode -> 
│   ├── DEF: offMode2outDist -> 
│   ├── DEF: offMode2outTime -> 
│   ├── DEF: offMode2outMode -> 
│   ├── DEF: offMode2Dist -> 
│   ├── DEF: offMode2Time -> 
│   ├── DEF: outTime2offTime
│   ├── DEF: cursor2offDist -> 
│   ├── DEF: cursor2offTime -> 
│   ├── DEF: cursor2outDist -> 
│   ├── DEF: cursor2outTime -> 
│   ├── DEF: outTime2cursor -> 
│   │   [TimeDistRef, lapTime2Dist, 
│   │   outTime2offTime]
│   ├── DEF: makeTD -> [TimeDistRef, 
│   │   duration, getTDValue, 
│   │   lapDist2Time, lapTime2Dist, 
│   │   makeTD]
│   ├── DEF: getTDValue
│   ├── DEF: getLapValue -> 
│   ├── DEF: windowSize2Mode -> 
│   ├── DEF: get_channel_prop -> 
│   │   [ChannelProperties]
│   ├── DEF: get_channel_data -> 
│   └── DEF: math_invalidate -> 
├── DEF: format_time -> 
└── DEF: atomic_write -> 
third_party/TrackDataAnalysis/ui/table_b
uilder.py
├── CLASS: ChannelSelect
│   ├── DEF: __init__ -> 
│   │   [QDialogButtonBox, QLineEdit, 
│   │   QListWidget, QVBoxLayout, 
│   │   __init__, addItems, addWidget, 
│   │   connect, fromhex, get, keys, 
│   │   restoreGeometry, 
│   │   setClearButtonEnabled, 
│   │   setLayout, setPlaceholderText, 
│   │   setWindowTitle, sorted, super]
│   ├── DEF: hideEvent -> 
│   └── DEF: text_changed -> 
│       [TextMatcher, count, isHidden, 
│       item, match, range, setHidden, 
│       text]
├── CLASS: ChannelEdit
│   ├── DEF: __init__ -> [__init__, 
│   │   addAction, connect, setReadOnly,
│   │   standardIcon, style, super]
│   └── DEF: selected -> [ChannelSelect,
│       exec_, len, selectedItems, 
│       setText, text]
├── CLASS: TBAxis
│   ├── DEF: __init__ -> [ChannelEdit, 
│   │   QFormLayout, QLineEdit, 
│   │   __init__, addRow, connect, get, 
│   │   setCheckable, setChecked, 
│   │   setLayout, setText, super]
│   ├── DEF: save_state -> 
│   └── DEF: recompute -> 
└── CLASS: TableBuilder
    ├── DEF: __init__ -> [ChannelEdit, 
    │   QFormLayout, QGroupBox, 
    │   QHBoxLayout, QTableWidget, 
    │   QVBoxLayout, TBAxis, __init__, 
    │   addLayout, addRow, addWidget, 
    │   connect, get, recompute, 
    │   setAlternatingRowColors, 
    │   setCheckable, setChecked, 
    │   setContextMenuPolicy, 
    │   setHorizontalScrollMode, 
    │   setLayout, setText, 
    │   setVerticalScrollMode, super]
    ├── DEF: save_state -> 
    ├── DEF: channels -> 
    ├── DEF: updateCursor
    ├── DEF: paintEvent -> [QColor, 
    │   fillRect, makePaintHelper]
    ├── DEF: clear_table -> 
    └── DEF: recompute -> 
        [QTableWidgetItem, argsort, 
        clear, clear_table, 
        column_stack, concatenate, full,
        get_channel_data, 
        horizontalHeader, interp_many, 
        isChecked, isinstance, len, 
        logical_or, nonzero, ones, 
        range, recompute, reduceat, 
        resizeSections, setColumnCount, 
        setHorizontalHeaderLabels, 
        setItem, setRowCount, 
        setTextAlignment, 
        setVerticalHeaderLabels, text, 
        unique, verticalHeader]
third_party/TrackDataAnalysis/ui/timedis
t.py
├── DEF: roundUpHumanNumber -> 
├── CLASS: AxisGrid
│   ├── DEF: calc
│   ├── DEF: invert
│   └── DEF: invertRelative
└── CLASS: TimeDist
    ├── DEF: __init__ -> 
    │   [MouseHelperClick, 
    │   MouseHelperItem, QAction, 
    │   QColor, QPen, __init__, 
    │   addAction, addMouseHelperTop, 
    │   bool, connect, setAcceptDrops, 
    │   setCheckable, setChecked, super]
    ├── DEF: save_state -> 
    ├── DEF: toggle_time_slip -> 
    ├── DEF: leftDrag -> 
    ├── DEF: rightDrag -> 
    ├── DEF: offsetCapture -> 
    ├── DEF: offsetDrag -> 
    ├── DEF: channelName -> 
    ├── DEF: graph_wheel -> 
    ├── DEF: cursorJump -> 
    ├── DEF: cursorDrag -> 
    ├── DEF: zoom_sel_start -> 
    ├── DEF: zoom_sel_drag -> 
    ├── DEF: zoom_sel_release -> 
    ├── DEF: xaxisCapture
    ├── DEF: xaxisDrag -> 
    ├── DEF: selectFont -> [QFont, 
    │   deviceScale, setPixelSize]
    ├── DEF: calc_time_slip -> 
    ├── DEF: paintGraph -> [AxisGrid, 
    │   MouseHelperClick, 
    │   MouseHelperItem, QColor, 
    │   QFontMetrics, QPen, QPoint, 
    │   QRect, QRectF, QSize, append, 
    │   arange, asarray, astype, 
    │   bisect_left, bisect_right, calc,
    │   ceil, channelName, 
    │   cursor2outTime, drawLine, 
    │   drawText, enumerate, fillRect, 
    │   getTDValue, get_channel_data, 
    │   get_channel_prop, get_laps, 
    │   height, interp, invert, left, 
    │   len, list, max, maximum, 
    │   memoryview, min, minimum, 
    │   offMode2outMode, paintYAxis, 
    │   reduceat, restore, right, round,
    │   roundUpHumanNumber, save, 
    │   selectFont, setClipRect, 
    │   setFont, setPen, setStyle, 
    │   subtract, unique, width, 
    │   windowSize2Mode, zip]
    ├── DEF: paintYAxis -> 
    │   [QFontMetrics, arange, calc, 
    │   drawLine, drawText, floor, 
    │   height, int, left, log10, max, 
    │   setFont, setPen, zip]
    ├── DEF: paintXAxis -> 
    ├── DEF: minmax_width -> 
    ├── DEF: paint_sectors -> [QColor, 
    │   bisect_right, calc, drawRect, 
    │   drawText, elidedText, fillRect, 
    │   find, fontMetrics, 
    │   horizontalAdvance, lapDist2Mode,
    │   lapMode2Dist, len, max, min, 
    │   range, round, width]
    ├── DEF: paintEvent -> [AxisGrid, 
    │   QBrush, QColor, QFontMetrics, 
    │   QPen, QPoint, QRect, calc, 
    │   calc_time_slip, channelName, 
    │   clear, drawLine, drawRect, 
    │   drawText, duration, enumerate, 
    │   fillRect, getLapValue, 
    │   getTDValue, get_channel_data, 
    │   get_laps, height, 
    │   horizontalAdvance, isChecked, 
    │   lapTime2Mode, len, lookupCursor,
    │   makePaintHelper, max, min, 
    │   minmax_width, outTime2Mode, 
    │   paintGraph, paintXAxis, 
    │   paint_sectors, range, restore, 
    │   round, roundUpHumanNumber, save,
    │   selectFont, setBackground, 
    │   setBackgroundMode, 
    │   setCompositionMode, setCoords, 
    │   setFont, setPen, setRect, 
    │   setStyle, setWidth, str, width, 
    │   windowSize2Mode]
    ├── DEF: tryRemoveChannel -> 
    ├── DEF: tryAddChannelExistingGroup 
    │   -> 
    ├── DEF: addChannel -> 
    ├── DEF: channels
    ├── DEF: updateCursor -> 
    ├── DEF: selectChannel -> 
    ├── DEF: moveChannel -> 
    ├── DEF: acceptable_drop -> 
    ├── DEF: dragEnterEvent -> 
    ├── DEF: dragMoveEvent -> 
    ├── DEF: dragLeaveEvent -> 
    ├── DEF: dropEvent -> 
    ├── DEF: update_time_slip_check -> 
    ├── DEF: channelMenuRemove -> 
    └── DEF: contextMenuEvent -> [QMenu,
        accept, actions, addAction, 
        addSeparator, channelMenuRemove,
        channel_editor, connect, exec_, 
        getLastMouseHelperData, 
        globalPos]
third_party/TrackDataAnalysis/ui/track.p
y
├── DEF: track_dir -> 
├── DEF: strcode -> 
├── DEF: strdeg -> 
├── DEF: coord_fname -> 
├── DEF: find_many_crossing -> 
├── DEF: load_track -> [Config, arange, 
│   array, column_stack, coord_fname, 
│   find_many_crossing, from_dict, len, 
│   list, lla2ecef, load, lower, max, 
│   min, open, perf_counter, print, 
│   scandir, sum, track_dir, zip]
├── DEF: save_track -> 
├── DEF: select_track -> [Marker, 
│   Sectors, Track, arctan2, array, 
│   concatenate, cos, cumsum, float, 
│   get_channel_data, 
│   get_key_channel_map, get_metadata, 
│   int, interp_many, len, linspace, 
│   list, lla2ecef, load_track, map, 
│   memoryview, nonzero, norm, 
│   perf_counter, pop, print, range, 
│   round, sin, zeros, zip]
├── CLASS: IndexDetails
├── CLASS: TrackTreeModel
│   ├── DEF: __init__ -> [__init__, 
│   │   super]
│   ├── DEF: child -> [IndexDetails, 
│   │   internalPointer, isValid, 
│   │   isinstance, keys, row, sorted]
│   ├── DEF: data -> 
│   ├── DEF: headerData
│   ├── DEF: index -> [QModelIndex, 
│   │   child, createIndex, hasIndex]
│   ├── DEF: parent -> [QModelIndex, 
│   │   createIndex, index, 
│   │   internalPointer, isValid, 
│   │   isinstance, items, sorted]
│   ├── DEF: rowCount -> 
│   └── DEF: columnCount
├── CLASS: TrackSectorsMapWidget
│   ├── DEF: __init__ -> [__init__, 
│   │   setMouseTracking, super]
│   ├── DEF: mouseMoveEvent -> 
│   ├── DEF: mousePressEvent -> 
│   ├── DEF: mouseReleaseEvent -> 
│   ├── DEF: crossing_vector -> 
│   └── DEF: paintEvent -> [QColor, 
│       QPen, QPoint, QRectF, append, 
│       array, ceil, column_stack, 
│       crossing_vector, deviceScale, 
│       drawLine, fillRect, 
│       find_crossing_idx, int, len, 
│       list, lla2ecef, makePaintHelper,
│       max, min, paint_satellite, 
│       range, row_stack, setPen, 
│       setWidth]
├── CLASS: TrackDialog
│   ├── DEF: __init__ -> [QComboBox, 
│   │   QDialogButtonBox, QFormLayout, 
│   │   QHBoxLayout, QLineEdit, 
│   │   QPushButton, QSplitter, 
│   │   QTreeView, QVBoxLayout, QWidget,
│   │   TrackSectorsMapWidget, 
│   │   TrackTreeModel, __init__, 
│   │   addRow, addWidget, connect, 
│   │   expandAll, fromhex, get, header,
│   │   labelForField, lineEdit, 
│   │   restoreGeometry, restoreState, 
│   │   selectionModel, setEditable, 
│   │   setEnabled, setLayout, setModel,
│   │   super]
│   ├── DEF: hideEvent -> 
│   ├── DEF: split_sector -> [Marker, 
│   │   array, emit, insert, interp, 
│   │   update, update_selection]
│   ├── DEF: remove_marker -> 
│   ├── DEF: marker_select -> 
│   ├── DEF: name_edit -> 
│   ├── DEF: type_edit -> 
│   ├── DEF: update_selection -> 
│   └── DEF: sector_click -> 
└── DEF: track_editor -> [TrackDialog, 
    deepcopy, emit, exec_, save_track]
third_party/TrackDataAnalysis/ui/video.p
y
├── CLASS: GetProcAddressGetter
│   ├── DEF: __init__ -> 
│   │   [AssertionError, 
│   │   QOffscreenSurface, create, 
│   │   create_window, currentContext, 
│   │   init, makeCurrent, 
│   │   make_context_current, 
│   │   window_hint]
│   └── DEF: wrap -> 
├── DEF: estimate_lap_offset -> 
│   [MP4_estimate_start_time, combine, 
│   date, fromisoformat, get_metadata, 
│   print_exc, timedelta, total_seconds]
├── CLASS: OneVideo
│   ├── DEF: __init__ -> [MPV, __init__,
│   │   connect, observe_property, 
│   │   setMouseTracking, setlocale, 
│   │   super, system]
│   ├── DEF: emit_time -> 
│   ├── DEF: emit_seeking -> 
│   ├── DEF: process_update -> 
│   ├── DEF: done_seeking -> 
│   ├── DEF: update_seeking -> 
│   ├── DEF: seek_cmd_done -> 
│   ├── DEF: play_cb -> 
│   ├── DEF: pause_cb -> 
│   ├── DEF: next_frame -> 
│   ├── DEF: prev_frame -> 
│   ├── DEF: process_result -> 
│   ├── DEF: mpv_command_async -> 
│   ├── DEF: async_idle -> 
│   ├── DEF: updateCursor -> 
│   ├── DEF: update_time -> 
│   ├── DEF: initializeGL -> 
│   │   [GetProcAddressGetter, 
│   │   MpvGlGetProcAddressFn, 
│   │   MpvRenderContext, updateCursor]
│   └── DEF: paintGL -> 
├── CLASS: AlignmentSlider
│   ├── DEF: __init__ -> 
│   │   [MouseHelperClick, 
│   │   MouseHelperItem, __init__, 
│   │   addMouseHelperTop, 
│   │   setMouseTracking, super]
│   ├── DEF: sizeHint -> [QFontMetrics, 
│   │   QSize, height, select_font]
│   ├── DEF: xaxis_capture
│   ├── DEF: xaxis_drag -> 
│   ├── DEF: select_font -> [QFont, 
│   │   deviceScale, setPixelSize]
│   ├── DEF: wheelEvent -> 
│   └── DEF: paintEvent -> [AxisGrid, 
│       QColor, QFontMetrics, QPen, abs,
│       calc, ceil, copysign, 
│       cursor2outTime, drawLine, 
│       drawText, floor, height, int, 
│       log10, makePaintHelper, range, 
│       roundUpHumanNumber, select_font,
│       setFont, setPen, setRect, 
│       setStyle, trunc, width]
└── CLASS: Video
    ├── DEF: __init__ -> 
    │   [AlignmentSlider, QAction, 
    │   QGridLayout, __init__, 
    │   addAction, addWidget, connect, 
    │   hide, setCheckable, 
    │   setContextMenuPolicy, setLayout,
    │   setRowStretch, setSeparator, 
    │   super, update_video_index]
    ├── DEF: save_state -> 
    ├── DEF: channels -> 
    ├── DEF: addChannel
    ├── DEF: update_video_index -> 
    │   [OneVideo, addWidget, 
    │   deleteLater, get_filename, 
    │   setParent, update_video_index]
    ├── DEF: updateCursor -> 
    ├── DEF: load_ref_video -> 
    ├── DEF: set_align_mode -> 
    ├── DEF: play_cb -> 
    ├── DEF: next_frame -> 
    └── DEF: prev_frame -> 
third_party/TrackDataAnalysis/ui/widgets
.py
├── DEF: deviceScaleFactor -> 
├── DEF: deviceScale -> 
├── DEF: devicePointScale -> 
├── DEF: makePaintHelper -> 
│   [PaintHelper, QPainter, QRectF, 
│   QSizeF, bottomRight, 
│   devicePixelRatioF, 
│   deviceScaleFactor, geometry, rect, 
│   scale, size, topLeft]
├── CLASS: RotatedPushButton
│   ├── DEF: paintEvent -> 
│   │   [QStyleOptionButton, 
│   │   QStylePainter, drawControl, 
│   │   height, initStyleOption, rotate,
│   │   translate, transposed]
│   └── DEF: sizeHint -> 
├── CLASS: MouseHelperClick
├── CLASS: MouseHelperItem
│   └── DEF: __init__ -> [QRectF]
├── CLASS: MouseHelperWidget
│   ├── DEF: __init__ -> [QPointF, 
│   │   __init__, setMouseTracking, 
│   │   super]
│   ├── DEF: lookupCursor -> [QCursor, 
│   │   contains, setCursor]
│   ├── DEF: addMouseHelperTop -> 
│   ├── DEF: addMouseHelperBottom -> 
│   ├── DEF: getLastMouseHelperData -> 
│   ├── DEF: getEventMouseHelperData -> 
│   ├── DEF: __handleClick -> 
│   ├── DEF: mousePressEvent -> 
│   │   [__handleClick, mousePressEvent,
│   │   super]
│   ├── DEF: mouseDoubleClickEvent -> 
│   │   [__handleClick, 
│   │   mouseDoubleClickEvent, super]
│   ├── DEF: mouseMoveEvent -> 
│   ├── DEF: mouseReleaseEvent -> 
│   └── DEF: wheelEvent -> 
└── CLASS: LapWidget
    ├── DEF: __init__ -> 
    │   [MouseHelperClick, 
    │   MouseHelperItem, __init__, 
    │   addMouseHelperBottom, connect, 
    │   super]
    ├── DEF: leftDrag -> 
    ├── DEF: rightDrag -> 
    ├── DEF: windowDragCapture
    ├── DEF: windowDrag -> 
    ├── DEF: selectLap -> [TimeDistRef, 
    │   emit, outMode2Time, x]
    ├── DEF: getFont -> [QFont, 
    │   deviceScale, setPixelSize]
    ├── DEF: updateCursor -> 
    ├── DEF: modeCalc -> 
    ├── DEF: timeCalc -> 
    ├── DEF: sizeHint -> [QFontMetrics, 
    │   QSize, devicePixelRatioF, 
    │   getFont, height]
    └── DEF: paintEvent -> [QColor, 
        QFontMetrics, QPen, QRect, 
        basename, chr, drawLine, 
        drawRect, drawText, duration, 
        enumerate, fillRect, 
        format_time, getFont, 
        get_filename, get_laps, 
        get_metadata, height, 
        horizontalAdvance, join, 
        lookupCursor, makePaintHelper, 
        max, modeCalc, outTime2Mode, 
        setCoords, setFont, setPen, 
        setRect, setStyle, str, 
        timeCalc, width]
third_party/TrackDataAnalysis/version.py
third_party/kn5-obj-converter/convert.py
├── DEF: cli -> [ArgumentParser, 
│   SystemExit, abspath, add_argument, 
│   convert_to_obj, endswith, exists, 
│   join, listdir, parse_args, print]
├── CLASS: kn5Material
│   └── DEF: __init__
├── CLASS: kn5Node
│   └── DEF: __init__ -> 
├── DEF: read_string -> 
├── DEF: matrix_mult -> 
├── DEF: matrix_to_euler -> 
├── DEF: scale_from_matrix -> 
├── DEF: read_nodes -> 
├── DEF: transparant_shader -> 
├── DEF: export_obj -> 
├── DEF: read_kn5 -> 
└── DEF: convert_to_obj -> 
third_party/xrk/test_xrk.py
└── CLASS: XrkTest
    ├── DEF: setUp -> [XRK]
    ├── DEF: tearDown -> 
    ├── DEF: testBasics -> 
    ├── DEF: testChannels -> 
    ├── DEF: testLapInfo -> 
    └── DEF: testTdLookup -> 
third_party/xrk/xrk.py
├── CLASS: TimeStruct
├── CLASS: XRKChannel
│   ├── DEF: __init__
│   ├── DEF: __repr__
│   ├── DEF: units -> 
│   └── DEF: samples -> 
├── CLASS: XRKGPSChannel
│   └── DEF: __init__ -> [__init__, 
│       super]
├── CLASS: XRKGPSrawChannel
│   └── DEF: __init__ -> [__init__, 
│       super]
└── CLASS: XRK
    ├── DEF: __init__ -> 
    ├── DEF: close -> 
    ├── DEF: __repr__
    ├── DEF: summary -> 
    ├── DEF: bestlap -> 
    ├── DEF: vehicle_name -> 
    ├── DEF: track_name -> 
    ├── DEF: racer_name -> 
    ├── DEF: championship_name -> 
    ├── DEF: venue_type -> 
    ├── DEF: datetime -> 
    ├── DEF: lapcount -> 
    ├── DEF: channels -> [XRKChannel, 
    │   XRKGPSChannel, XRKGPSrawChannel,
    │   decode, get_GPS_channel_name, 
    │   get_GPS_channels_count, 
    │   get_GPS_raw_channel_name, 
    │   get_GPS_raw_channels_count, 
    │   get_channel_name, 
    │   get_channels_count, range]
    ├── DEF: timedistance -> 
    ├── DEF: _tdlookup -> 
    ├── DEF: timetodistance -> 
    │   [_tdlookup]
    ├── DEF: distancetotime -> 
    │   [_tdlookup]
    └── DEF: lap_info -> 
track/__init__.py
track/constants.py
track/data/RIMSportKarting/export/alfano
/decode_alfano.py
├── CLASS: AlfanoTrack
│   ├── DEF: __init__
│   ├── DEF: from_file -> [Path, 
│   │   from_bytes, read_bytes]
│   ├── DEF: from_bytes -> 
│   │   [_parse_header, _parse_points, 
│   │   bytes, cls]
│   ├── DEF: _parse_header -> 
│   │   [ValueError, decode, from_bytes,
│   │   hex, replace, startswith, strip,
│   │   unpack_from]
│   ├── DEF: _parse_points -> 
│   ├── DEF: to_bytes -> 
│   ├── DEF: save_trackALFANO -> [Path, 
│   │   to_bytes, write_bytes]
│   ├── DEF: save_geojson -> [Path, 
│   │   append, dumps, get, len, 
│   │   write_text]
│   ├── DEF: save_csv -> [DictWriter, 
│   │   open, writeheader, writerows]
│   ├── DEF: apexes
│   └── DEF: stats -> 
└── DEF: main -> [Path, from_file, 
    items, len, min, print, range, 
    read_bytes, save_csv, save_geojson, 
    save_trackALFANO, stat, stats, 
    unlink]
track/data/RIMSportKarting/export/motec/
convert_vn1.py
├── DEF: geojson_to_vn1 -> [VN1Encoder, 
│   VN1Metadata, VN1Venue, ValueError, 
│   append, encode, get, len, load, 
│   open, print, read]
├── DEF: vn1_to_geojson -> 
├── DEF: vn1_to_csv -> 
├── DEF: csv_to_vn1 -> [VN1Encoder, 
│   VN1Metadata, VN1Venue, ValueError, 
│   append, encode, float, len, open, 
│   print, read, readline, split, strip]
├── DEF: simplify_coordinates -> 
└── DEF: point_line_distance -> 
track/data/RIMSportKarting/export/motec/
parse_vn1.py
├── CLASS: VN1Venue
├── DEF: parse_vn1 -> [VN1Venue, append,
│   decode, len, open, read, rstrip, 
│   unpack]
├── DEF: print_summary -> 
└── DEF: analyze_metadata -> 
track/data/RIMSportKarting/export/motec/
test_vn1_codec.py
├── DEF: test_decode -> [Path, abs, 
│   decode_vn1, len, print, str]
├── DEF: test_encode -> 
│   [NamedTemporaryFile, Path, 
│   VN1Encoder, decode_vn1, encode, len,
│   print, unlink]
├── DEF: test_roundtrip -> 
│   [NamedTemporaryFile, Path, 
│   VN1Encoder, abs, decode_vn1, encode,
│   len, print, str, unlink, zip]
├── DEF: test_geojson_conversion -> 
│   [NamedTemporaryFile, Path, exists, 
│   len, load, open, print, unlink, 
│   vn1_to_geojson]
├── DEF: test_csv_conversion -> 
│   [NamedTemporaryFile, Path, exists, 
│   len, open, print, readlines, strip, 
│   unlink, vn1_to_csv]
├── DEF: test_bounds -> [Path, 
│   decode_vn1, max, min, print, str]
└── DEF: run_all_tests -> 
track/data/RIMSportKarting/export/motec/
vn1_codec.py
├── CLASS: VN1Metadata
├── CLASS: VN1Venue
├── CLASS: VN1Decoder
│   ├── DEF: __init__ -> [Path, 
│   │   _read_file]
│   ├── DEF: _read_file -> 
│   ├── DEF: decode -> [VN1Metadata, 
│   │   VN1Venue, ValueError, 
│   │   _parse_coordinates, 
│   │   _parse_string, len, print, 
│   │   unpack]
│   ├── DEF: _parse_string -> 
│   └── DEF: _parse_coordinates -> 
├── CLASS: VN1Encoder
│   ├── DEF: __init__
│   ├── DEF: encode -> 
│   │   [_write_coordinates, 
│   │   _write_string, bytearray, int, 
│   │   len, open, pack_into, write]
│   ├── DEF: _write_string -> 
│   └── DEF: _write_coordinates -> 
├── DEF: print_venue_info -> 
├── DEF: decode_vn1 -> [VN1Decoder, 
│   decode]
└── DEF: encode_vn1 -> [VN1Encoder, 
    encode]
track/segmentation.py
├── DEF: segment_track -> 
├── DEF: create_sectors_from_distances 
│   -> 
├── DEF: load_sectors_json -> [Path, 
│   get, is_file, isinstance, len, load,
│   open, sorted]
└── DEF: find_segment_by_distance -> 
track/track.py
├── DEF: _determine_utm_zone_from_coords
│   -> 
└── CLASS: Track
    ├── DEF: __init__ -> 
    │   [_determine_utm_zone_from_coords
    │   , column_stack, get_transformer,
    │   len, transform]
    ├── DEF: get_transformer -> 
    ├── DEF: _get_transformer_to_wgs84 
    │   -> 
    ├── DEF: bounds -> 
    ├── DEF: bounds
    ├── DEF: start_finish_wgs84 -> 
    │   [_get_transformer_to_wgs84, 
    │   array, list, transform, zip]
    ├── DEF: bestline_wgs84 -> 
    │   [_get_transformer_to_wgs84, 
    │   array, list, transform, zip]
    ├── DEF: _init_projector -> 
    ├── DEF: total_length -> 
    │   [_init_projector]
    ├── DEF: has_projector -> 
    │   [_init_projector, len]
    ├── DEF: layout
    ├── DEF: geometry
    ├── DEF: centerline -> 
    ├── DEF: start_finish_intersection 
    │   -> [LineString, array, 
    │   interpolate, intersection, len, 
    │   project, tuple]
    ├── DEF: project -> 
    │   [_init_projector, argmin, clip, 
    │   errstate, float, nan_to_num, 
    │   sqrt, sum]
    ├── DEF: load -> [Path, ValueError, 
    │   _determine_utm_zone_from_coords,
    │   abs, append, arctan2, array, 
    │   cls, column_stack, 
    │   compute_centerline, debug, 
    │   degrees, exists, get, 
    │   get_transformer, info, join, 
    │   len, list, 
    │   load_polyline_geojson, 
    │   load_track_config, map, mean, 
    │   median, min, range, 
    │   resample_linestring, 
    │   segment_track, transform, 
    │   transform_coordinates, 
    │   validate_track_directory, 
    │   warning, zip]
    ├── DEF: load_bestline -> [Path, 
    │   array, debug, exists, 
    │   get_transformer, info, len, 
    │   list, load_polyline_geojson, 
    │   map, transform]
    ├── DEF: set_bestline_from_gps -> 
    └── DEF: save_bestline -> [Path, 
        append, array, dump, float, 
        get_transformer, info, mkdir, 
        open, transform, warning, zip]
track/utils.py
├── DEF: normalize_angle
├── DEF: calculate_heading -> 
├── DEF: load_polyline_geojson -> 
├── DEF: load_track_config -> [Path, 
│   is_file, load, open]
├── DEF: get_transformer -> 
├── DEF: transform_coordinates -> 
├── DEF: resample_linestring -> 
└── DEF: compute_centerline -> 
track/validation.py
├── DEF: validate_geojson_polyline -> 
├── DEF: validate_geojson_crs -> 
└── DEF: validate_track_directory -> 
    [Path, append, exists, is_dir, 
    items, len, load, open, 
    validate_geojson_polyline]
track/visualize_track.py
├── DEF: plot_track -> [LineString, 
│   ValueError, add_basemap, annotate, 
│   array, arrow, close, debug, dict, 
│   enumerate, exists, flatten, 
│   get_transformer, info, legend, len, 
│   load_polyline_geojson, max, min, 
│   plot, reshape, savefig, set_aspect, 
│   set_facecolor, set_title, set_xlim, 
│   set_xticks, set_ylim, set_yticks, 
│   show, subplots, text, tight_layout, 
│   transform_coordinates, warning]
└── DEF: main -> [ArgumentParser, Path, 
    add_argument, error, exception, 
    exists, exit, format_exc, info, 
    is_dir, load, parse_args, 
    plot_track, tuple]
utils/__init__.py
utils/ass.py
├── CLASS: AssBuilder
│   ├── DEF: __init__ -> 
│   ├── DEF: add_style -> 
│   ├── DEF: add_event -> 
│   ├── DEF: write -> 
│   └── DEF: write_with_offset -> 
├── DEF: fmt_ass_time -> 
├── DEF: emit_lap_stats_ass -> 
└── DEF: emit_gauge_ass -> 
    [PredictiveLapModel, abs, add_event,
    add_style, ceil, copy, delta_to_x, 
    enumerate, fillna, fmt_ass_time, 
    fmt_lap_time, get, get_lap_stats, 
    get_time, getattr, groupby, int, 
    isnan, items, keys, len, list, max, 
    min, next, print, range, sorted, 
    sum, to_dict, zeros, zip]
utils/cuda.py
└── DEF: check_cuda_availability -> 
utils/generate_report.py
├── DEF: get_font -> 
├── DEF: format_time -> 
├── DEF: format_speed -> 
├── DEF: calculate_sector_stats -> 
├── DEF: draw_track_map -> 
├── DEF: draw_lap_table -> 
├── DEF: draw_statistics -> 
├── DEF: calculate_acceleration_stats ->
├── DEF: generate_report -> [Draw, 
│   add_lap_numbers, 
│   calculate_sector_stats, 
│   detect_crossings, draw_lap_table, 
│   draw_statistics, draw_track_map, 
│   get_font, get_lap_stats, len, load, 
│   max, min, new, print, range, 
│   rectangle, save, text]
└── DEF: main -> [ArgumentParser, Path, 
    add_argument, generate_report, 
    parse_args]
utils/sync_ui.py
├── DEF: run_interactive_sync -> 
│   [VideoCapture, append, 
│   destroyAllWindows, get, imshow, int,
│   isOpened, items, len, max, min, 
│   namedWindow, ord, print, putText, 
│   read, rectangle, release, 
│   resizeWindow, set, str, sum, 
│   waitKey]
├── DEF: run_manual_lap_marking -> 
│   [VideoCapture, abs, append, 
│   destroyAllWindows, enumerate, get, 
│   imshow, int, isOpened, len, max, 
│   min, namedWindow, ord, print, 
│   putText, read, rectangle, release, 
│   resizeWindow, set, sorted, str, 
│   waitKey]
└── DEF: run_trim_selection -> 
    [VideoCapture, addWeighted, copy, 
    destroyAllWindows, get, imshow, int,
    isOpened, max, min, namedWindow, 
    ord, print, putText, range, read, 
    rectangle, release, resizeWindow, 
    set, str, waitKey]
utils/telemetry_sync.py
├── DEF: _align_crossings -> 
└── DEF: create_session_from_crossings 
    -> [DataFrame, Session, arange, 
    array, bisect_right, get_lap_number,
    sorted]
utils/video_pipeline.py
├── CLASS: Pipeline
├── DEF: write_pgm_u16 -> [ValueError, 
│   byteswap, encode, tobytes, 
│   write_bytes]
├── DEF: _compute_undistort_maps -> 
├── DEF: _create_remap_arrays -> 
├── DEF: make_fisheye_remap_maps -> 
│   [_compute_undistort_maps, 
│   _create_remap_arrays, write_pgm_u16]
├── DEF: build_opener -> [Pipeline, 
│   input, str]
├── DEF: build_trimer -> [Pipeline, 
│   filter]
├── DEF: _load_remap_stream -> 
├── DEF: build_undistorter -> [Path, 
│   Pipeline, _load_remap_stream, close,
│   make_fisheye_remap_maps, mkstemp, 
│   register, remap, unlink]
├── DEF: build_transform_estimator -> 
│   [Pipeline, filter, str, videohint]
├── DEF: build_stabilizer -> [Pipeline, 
│   filter, str, videohint]
├── DEF: build_ov -> [Pipeline, overlay]
└── DEF: build_writer -> 
video/__init__.py
video/concat/concat.py
├── CLASS: VideoData
├── DEF: check_system_dependencies -> 
├── DEF: extract_frame -> [BytesIO, 
│   input, open, output, run, str]
├── DEF: parse_timestamp -> 
├── DEF: detect_timestamp_from_image -> 
├── DEF: sample_timestamps -> 
├── DEF: estimate_start_time -> 
│   [Counter, append, date, len, 
│   most_common, sort, timedelta]
├── DEF: get_video_files -> 
├── DEF: analyze_video -> [VideoData, 
│   estimate_start_time, probe_video, 
│   sample_timestamps, timedelta]
├── DEF: check_theoretical_continuity ->
├── DEF: check_explicit_continuity -> 
├── DEF: check_gap_filling -> 
├── DEF: check_date_correction -> 
├── DEF: is_continuous -> 
├── DEF: group_videos -> 
├── DEF: export_group -> 
└── DEF: main -> [ArgumentParser, 
    Progress, SpinnerColumn, TextColumn,
    add_argument, add_task, advance, 
    analyze_video, append, 
    check_system_dependencies, 
    enumerate, exists, exit, 
    export_group, get_video_files, 
    group_videos, len, mkdir, 
    parse_args, print, strftime, update]
video/concat/debug_crop.py
└── DEF: debug_crop -> [Draw, copy, 
    crop, detect_timestamp_from_image, 
    exists, extract_frame, int, mkdir, 
    print, rectangle, save]
video/concat/test_robustness.py
├── DEF: split_video -> 
├── DEF: temp_dir -> [Path, exists, 
│   mkdir, resolve, rmtree]
└── DEF: test_all_videos_robustness -> 
    [Path, abs, analyze_video, append, 
    glob, group_videos, iterdir, len, 
    print, range, skip, sorted, 
    split_video, total_seconds, unlink, 
    zip]
video/split.py
├── DEF: parse_time_to_seconds -> 
│   [ValueError, float, int, len, split]
├── DEF: get_video_duration -> 
├── DEF: split_video -> 
│   [FileNotFoundError, Path, 
│   ValueError, exists, 
│   get_video_duration, mkdir, print, 
│   run, stat, str]
└── DEF: main -> [ArgumentParser, 
    add_argument, exit, parse_args, 
    parse_time_to_seconds, print, 
    split_video]
video/stab.py
├── DEF: generate_transforms -> 
├── DEF: get_transform_filter -> 
├── DEF: stabilize_video -> 
└── DEF: main -> [ArgumentParser, 
    add_argument, exists, exit, 
    expanduser, parse_args, print, 
    resolve, stabilize_video, with_name]
video/transcode.py
├── DEF: tqdm
├── DEF: get_video_files -> 
├── DEF: build_av1_nvenc_args -> 
├── DEF: build_hevc_nvenc_args -> 
├── DEF: build_svtav1_args -> 
├── DEF: build_x265_args -> 
├── DEF: get_video_codec_args -> 
├── DEF: build_ffmpeg_command -> 
├── DEF: transcode_file -> 
├── DEF: transcode_single_file -> 
├── DEF: transcode_multiple_files -> 
├── DEF: parse_args -> [ArgumentParser, 
│   add_argument, parse_args]
└── DEF: main -> 
video/trim.py
├── CLASS: VideoSidecar
│   ├── DEF: info_path
│   ├── DEF: load -> [Path, cls, exists,
│   │   loads, print, read_text]
│   ├── DEF: save -> 
│   └── DEF: get -> 
├── CLASS: TrimInfo
│   ├── DEF: info_path -> 
│   ├── DEF: load -> 
│   └── DEF: save -> 
├── CLASS: CrossingsInfo
│   ├── DEF: __post_init__
│   ├── DEF: info_path
│   ├── DEF: load -> [Path, bool, cls, 
│   │   exists, get, loads, print, 
│   │   read_text]
│   └── DEF: save -> 
├── DEF: parse_args -> [ArgumentParser, 
│   add_argument, parse_args]
├── DEF: get_crossings_info -> 
├── DEF: get_trim_info -> 
└── DEF: main -> [Path, exists, exit, 
    expanduser, float, get_trim_info, 
    group, input, output, 
    overwrite_output, parse_args, print,
    probe, replace, resolve, run, 
    search, str, strftime, strptime, 
    timedelta, with_name]
video/undistort.py
├── DEF: compute_maps -> 
├── DEF: worker_thread_func -> 
├── DEF: writer_thread_func -> 
└── DEF: main -> [ArgumentParser, Event,
    Queue, Thread, VideoCapture, 
    abspath, add_argument, append, 
    astype, basicConfig, boxFilter, 
    close, compile, compute_maps, 
    copyTo, cuda_GpuMat, 
    destroyAllWindows, error, exists, 
    exit, full, get, 
    getCudaEnabledDeviceCount, imshow, 
    info, input, int, isOpened, is_set, 
    join, output, overwrite_output, 
    parse_args, probe, put, range, read,
    release, remap, resize, run_async, 
    set, splitext, start, upload, wait, 
    waitKey, warning]
video/video_info.py
├── CLASS: VideoInfo
├── DEF: to_float -> 
└── DEF: probe_video -> [RuntimeError, 
    VideoInfo, decode, get, int, max, 
    partition, probe, str, to_float]
