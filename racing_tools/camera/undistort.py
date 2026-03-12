#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import csv
import sys
import os
import logging
import time
import threading
import queue
import ffmpeg

from racing_tools.camera.model import CameraModel

def compute_maps(intrinsics_path, w, h, balance, fov_scale):
    camera = CameraModel.load(intrinsics_path)
    K = camera.matrix
    D = camera.dist_coeffs
    model = camera.model_name
    dim = (w, h)
    
    if "Fisheye" in model:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, dim, np.eye(3), balance=balance, fov_scale=fov_scale
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2) 
    else:
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, dim, balance, dim, 0)
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, dim, cv2.CV_16SC2)
        
    return map1, map2

# Global maps for threads (read-only)
g_map1 = None
g_map2 = None
g_mask = None
g_invalid = None

# CUDA Globals
g_cuda_map1 = None
g_cuda_map2 = None
g_cuda_stream = None

def worker_thread_func(input_q, output_q, border_mode):
    global g_map1, g_map2, g_mask, g_invalid
    
    while True:
        try:
            item = input_q.get(timeout=1.0)
        except queue.Empty:
            continue
            
        if item is None:
            break
        
        idx, frame = item

        if g_cuda_map1 is not None:
             try:
                # Upload
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                # Remap on GPU
                # borderMode and interpolation map 1:1 usually
                # replicate is BORDER_REPLICATE
                # blur mode uses REFLECT_101 then CPU post-proc? Or GPU?
                
                if border_mode == 'blur':
                     # GPU REMAP REFLECT
                     gpu_res = cv2.cuda.remap(gpu_frame, g_cuda_map1, g_cuda_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                     
                     # Download for CPU post-processing (blurring on GPU is possible but let's keep it simple or user specific?)
                     # User wants optimize.
                     res = gpu_res.download()
                     
                     # Continue with CPU blur optimization for now, unless we do gpu blur?
                     # Let's stick to CPU post-proc for the blur part as logic matches CPU path
                     blurred = cv2.boxFilter(res, -1, (41, 41))
                     blurred = (blurred * 0.6).astype(np.uint8)
                     cv2.copyTo(blurred, g_invalid, res)
                     
                else:
                    mode = cv2.BORDER_CONSTANT
                    if border_mode == 'replicate':
                        mode = cv2.BORDER_REPLICATE
                        
                    gpu_res = cv2.cuda.remap(gpu_frame, g_cuda_map1, g_cuda_map2, interpolation=cv2.INTER_LINEAR, borderMode=mode)
                    res = gpu_res.download()
                    
                output_q.put((idx, res))
                continue
                
             except Exception as e:
                 logging.error(f"CUDA Error: {e}")
                 # Fallback to CPU? or fail?
                 output_q.put((idx, None))
                 continue
        
        try:
            if border_mode == 'replicate':
                res = cv2.remap(frame, g_map1, g_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            elif border_mode == 'blur':
                # Single Remap Optimization
                # 1. Remap with Mirror Reflection (covers both valid content and extended background)
                res = cv2.remap(frame, g_map1, g_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                
                # 2. Blur the whole thing (fast box filter)
                blurred = cv2.boxFilter(res, -1, (41, 41))
                blurred = (blurred * 0.6).astype(np.uint8) # Darken
                
                # 3. Composite ONLY in the invalid region using cv2.copyTo
                cv2.copyTo(blurred, g_invalid, res)

            else: # black
                res = cv2.remap(frame, g_map1, g_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            
            output_q.put((idx, res))
            
        except Exception as e:
            logging.error(f"Error in worker: {e}")
            output_q.put((idx, None)) # Signal failure for this frame? OR None to stop?
            # Better to skip or retry? 
            # For now, let's put a blank frame or re-raise?
            # Let's just print and continue?
            pass

def writer_thread_func(output_q, ffmpeg_stdin, total_frames, stop_event):
    next_idx = 0
    buffer = {}
    
    while next_idx < total_frames and not stop_event.is_set():
        try:
            item = output_q.get(timeout=0.5)
        except queue.Empty:
            continue
            
        idx, frame = item
        buffer[idx] = frame
        
        while next_idx in buffer:
            frame_to_write = buffer.pop(next_idx)
            if frame_to_write is None:
                # Error placeholder
                next_idx += 1
                continue
                
            try:
                ffmpeg_stdin.write(frame_to_write.tobytes())
            except BrokenPipeError:
                logging.error("FFmpeg stdin broken pipe.")
                stop_event.set()
                return
            next_idx += 1
            
            if next_idx % 100 == 0:
                 sys.stdout.write(f"\rProcessed {next_idx}/{total_frames} frames")
                 sys.stdout.flush()

    logging.info("\nWriter thread finished.")

def main():
    global g_map1, g_map2, g_mask, g_invalid, g_cuda_map1, g_cuda_map2

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description='Undistort video using OpenCV (Threading) and pipe to FFmpeg.')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('intrinsics', help='Path to intrinsics CSV file')
    parser.add_argument('--output', default=None, help='Path to output video file')
    parser.add_argument('--lossless', default=False, action='store_true', help='Use lossless output.')
    parser.add_argument('--balance', type=float, default=1.0, help='Balance 0.0-1.0')
    parser.add_argument('--fov_scale', type=float, default=1.15, help='FOV Scale (default 0.7 to avoid cropping)')
    parser.add_argument('--preview', action='store_true', help='Show a preview frame')
    parser.add_argument('--border_mode', default='blur', choices=['black', 'replicate', 'blur'], help='Border mode')
    parser.add_argument('--codec', default='h264', choices=['h264', 'av1', 'libx264'], help='Video codec: h264 (h264_nvenc), av1 (av1_nvenc), libx264 (CPU)')
    parser.add_argument('--threads', type=int, default=4, help='Number of worker threads (default: 4, typically sufficient if GIL released)')

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_video)
    if not os.path.exists(input_path):
        logging.error(f"Input video not found: {input_path}")
        sys.exit(1)

    if args.output is None:
        base, ext = os.path.splitext(input_path)
        args.output = f"{base}_undistorted.mp4"

    # --- Video Info ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logging.error("Failed to open video")
        sys.exit(1)
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logging.info(f"Video: {w}x{h} @ {fps}fps, {total_frames} frames")
    
    # --- Precompute Maps & Mask (Global) ---
    logging.info("Precomputing maps...")
    g_map1, g_map2 = compute_maps(args.intrinsics, w, h, args.balance, args.fov_scale)
    
    # helper to check cuda
    use_cuda = False
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            use_cuda = True
            logging.info(f"CUDA capable devices: {count}. Using cv2.cuda.remap.")
    except AttributeError:
        logging.info("OpenCV not built with CUDA support.")

    if use_cuda:
        global g_cuda_map1, g_cuda_map2
        g_cuda_map1 = cv2.cuda_GpuMat()
        g_cuda_map2 = cv2.cuda_GpuMat()
        g_cuda_map1.upload(g_map1)
        g_cuda_map2.upload(g_map2)
    
    if args.border_mode == 'blur':
        logging.info("Precomputing mask for blur mode...")
        # Map a white image to find valid pixels
        dummy = np.full((h, w), 255, dtype=np.uint8)
        # Using INTER_NEAREST for exact mask
        g_mask = cv2.remap(dummy, g_map1, g_map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # g_mask is 255 for valid, 0 for invalid
        g_invalid = ((g_mask == 0).astype(np.uint8)) * 255

    # --- Preview Mode ---
    if args.preview:
        logging.info("Showing preview...")
        if total_frames > 100:
             cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logging.error("Failed to read preview frame")
            sys.exit(1)
            
        if args.border_mode == 'blur':
            res = cv2.remap(frame, g_map1, g_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            blurred = cv2.boxFilter(res, -1, (41, 41))
            blurred = (blurred * 0.6).astype(np.uint8)
            cv2.copyTo(blurred, g_invalid, res)
        elif args.border_mode == 'replicate':
             res = cv2.remap(frame, g_map1, g_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
             res = cv2.remap(frame, g_map1, g_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        display_h = 720
        scale = display_h / h
        display_w = int(w * scale)
        preview_small = cv2.resize(res, (display_w, display_h))
        cv2.imshow("Preview", preview_small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit(0)

    cap.release() 
    cap = cv2.VideoCapture(input_path) 

    # --- Output Setup ---
    # Construct ffmpeg command using ffmpeg-python
    input_video = ffmpeg.input(
        'pipe:', 
        format='rawvideo', 
        pix_fmt='bgr24', 
        s=f'{w}x{h}', 
        r=fps,
        thread_queue_size=512
    )

    # Check for audio stream
    has_audio = False
    try:
        probe = ffmpeg.probe(input_path)
        for stream in probe['streams']:
            if stream['codec_type'] == 'audio':
                has_audio = True
                break
    except ffmpeg.Error as e:
        logging.warning(f"Could not probe input for audio: {e.stderr}")
    
    input_audio = None
    if has_audio:
        input_audio = ffmpeg.input(input_path, thread_queue_size=512)

    output_args = {
        'pix_fmt': 'yuv420p',
    }

    # Select codec
    if args.codec == 'av1':
        output_args['vcodec'] = 'av1_nvenc'
        output_args['preset'] = 'p7'
        output_args['tune'] = 'hq'
        if args.lossless:
            output_args['rc'] = 'constqp'
            output_args['qp'] = '0'
        else:
            output_args['rc'] = 'vbr'
            output_args['cq'] = '20'  # AV1 CQ 20 = visually lossless
            output_args['multipass'] = '2'
            output_args['maxrate'] = '40M'
            output_args['bufsize'] = '80M'
            output_args['rc-lookahead'] = '64'
            output_args['spatial_aq'] = '1'
            output_args['temporal_aq'] = '1'
    elif args.codec == 'h264':
        output_args['vcodec'] = 'h264_nvenc'
        output_args['preset'] = 'p7'
        output_args['tune'] = 'hq'
        if args.lossless:
             output_args['rc'] = 'constqp'
             output_args['qp'] = '0'
        else:
             output_args['rc'] = 'vbr'
             output_args['cq'] = '19'
             output_args['multipass'] = '2'
             output_args['maxrate'] = '50M'
             output_args['bufsize'] = '100M'
    else:  # libx264
        output_args['vcodec'] = 'libx264'
        if args.lossless:
            output_args['preset'] = 'ultrafast'
            output_args['qp'] = '0'
        else:
            output_args['crf'] = '18'
            output_args['preset'] = 'fast'

    inputs = [input_video]
    if has_audio:
        # Map audio from file: '1:a' (or 0:a if it was first input, but video pipe is 0)
        # The graph inputs are inputs[0] and inputs[1].
        # We can just pass input_audio.audio to output()
        inputs.append(input_audio.audio)

    # Note: ffmpeg-python automatically handles mapping if we pass stream objects.
    output_stream = ffmpeg.output(
        *inputs,
        args.output,
        **output_args
    ).overwrite_output()

    logging.info(f"FFmpeg Command: {' '.join(ffmpeg.compile(output_stream))}")
    
    process = output_stream.run_async(pipe_stdin=True)

    # --- Threading Setup ---
    # Python threads share memory, so we pass image objects around without copy!
    # Global maps (g_map1, g_map2) are read-only and shared.
    
    # We buffer input to keep workers busy.
    input_q = queue.Queue(maxsize=128) 
    output_q = queue.Queue(maxsize=128)
    
    num_threads = args.threads
    logging.info(f"Starting {num_threads} worker threads...")
    
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker_thread_func, args=(input_q, output_q, args.border_mode))
        t.start()
        threads.append(t)
        
    stop_event = threading.Event()
    writer_thread = threading.Thread(target=writer_thread_func, args=(output_q, process.stdin, total_frames, stop_event))
    writer_thread.start()
    
    # --- Main Loop (Producer) ---
    try:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if stop_event.is_set():
                break
                
            # Block if full
            input_q.put((idx, frame)) 
            idx += 1
            
    except KeyboardInterrupt:
        logging.info("Interrupted. Stopping...")
        stop_event.set()
        
    finally:
        logging.info("Signaling threads to stop...")
        # Signal workers
        for _ in range(num_threads):
            input_q.put(None)
            
        logging.info("Waiting for workers...")
        for t in threads:
            t.join()
            
        logging.info("Waiting for writer...")
        stop_event.set() # Just in case writer is stuck? Writer waits on output_q
        # If output_q is empty and writer waits, it has timeout.
        writer_thread.join()
        
        if process.stdin:
            process.stdin.close()
        process.wait()

        cap.release()
        logging.info("Done.")

if __name__ == '__main__':
    main()
