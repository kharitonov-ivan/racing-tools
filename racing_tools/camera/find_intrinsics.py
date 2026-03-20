import argparse
import cv2
import numpy as np
import csv
import os
import sys
import logging


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Find camera intrinsics from video with checkerboard.")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("--output", default=None, help="Path to output CSV file (default: intrinsics.csv in video directory)")
    parser.add_argument("--rows", type=int, default=6, help="Number of inner corners rows")
    parser.add_argument("--cols", type=int, default=9, help="Number of inner corners columns")
    parser.add_argument("--size", type=float, default=25.0, help="Size of each square in mm (default: 25.0)")
    parser.add_argument("--sample_interval", type=int, default=15, help="Frame interval to sample for calibration")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for OpenCV (cv2.setNumThreads)")
    parser.add_argument("--enforce_symmetry", action="store_true", help="Augment data by mirroring points to enforce symmetric calibration")
    parser.add_argument("--show", action="store_true", help="Show detected corners during processing")

    args = parser.parse_args()

    # Determine default output path if not specified
    if args.output is None:
        video_dir = os.path.dirname(args.input_video)
        args.output = os.path.join(video_dir, "intrinsics.csv")

    # Set OpenCV threads
    cv2.setNumThreads(args.threads)

    logging.info(f"Starting {sys.argv[0]}")
    logging.info(
        f"Arguments: input_video='{args.input_video}', rows={args.rows}, cols={args.cols}, size={args.size}, sample_interval={args.sample_interval}, threads={args.threads}, output='{args.output}'"
    )

    if not os.path.exists(args.input_video):
        logging.error(f"Video file '{args.input_video}' not found.")
        sys.exit(1)

    # Checkerboard dimensions
    CHECKERBOARD = (args.rows, args.cols)

    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)

    # Scale object points by square size
    objp *= args.size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        logging.error(f"Could not open video '{args.input_video}'.")
        sys.exit(1)

    frame_count = 0
    detected_count = 0

    logging.info(f"Processing video: {args.input_video}")
    logging.info(f"Searching for checkerboard pattern {CHECKERBOARD} every {args.sample_interval} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % args.sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_corners, corners = cv2.findChessboardCorners(
                gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if ret_corners:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                detected_count += 1
                logging.info(f"Frame {frame_count}: Pattern found! (Total: {detected_count})")

                if args.show:
                    cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret_corners)
                    cv2.imshow("img", frame)
                    cv2.waitKey(1)
            else:
                # Optional: print specific frames where it failed if in debug mode, otherwise keep it clean
                pass

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    if detected_count < 10:
        logging.warning(f"Only {detected_count} valid frames found. Calibration might be poor.")
        if detected_count == 0:
            logging.error("No checkerboard patterns found. Check rows/cols or video quality.")
            sys.exit(1)

            sys.exit(1)

    # We grab the image size from the last processed frame (gray)
    # Needed for symmetry augmentation and calibration
    h, w = gray.shape[:2]
    img_size = (w, h)

    # Enforce symmetry by data augmentation
    if args.enforce_symmetry:
        logging.info("Enforcing symmetry: Augmenting data with mirrored points...")
        augmented_objpoints = []
        augmented_imgpoints = []

        for ob, im in zip(objpoints, imgpoints):
            # Original
            augmented_objpoints.append(ob)
            augmented_imgpoints.append(im)

            # Flip Horizontally: x' = w - 1 - x
            # Note: The object points for the mirrored image should technically be the same 3D structure,
            # but viewed from a mirrored pose. We reuse the same objp because the physical board doesn't change,
            # we just simulate a camera flip.
            im_h = im.copy()
            im_h[:, :, 0] = (w - 1) - im_h[:, :, 0]
            augmented_objpoints.append(ob)
            augmented_imgpoints.append(im_h)

            # Flip Vertically: y' = h - 1 - y
            im_v = im.copy()
            im_v[:, :, 1] = (h - 1) - im_v[:, :, 1]
            augmented_objpoints.append(ob)
            augmented_imgpoints.append(im_v)

            # Flip Both
            im_hv = im.copy()
            im_hv[:, :, 0] = (w - 1) - im_hv[:, :, 0]
            im_hv[:, :, 1] = (h - 1) - im_hv[:, :, 1]
            augmented_objpoints.append(ob)
            augmented_imgpoints.append(im_hv)

        objpoints = augmented_objpoints
        imgpoints = augmented_imgpoints
        detected_count = len(objpoints)
        logging.info(f"Data augmented. Total patterns for calibration: {detected_count}")

    logging.info(f"Calibrating camera with {detected_count} frames...")

    # --- STANDARD PINHOLE CALIBRATION ---
    logging.info("--- Standard Pinhole Calibration ---")
    logging.info(f"Running cv2.calibrateCamera with {len(objpoints)} patterns... (this may take a while)")
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    logging.info("Standard calibration finished.")

    logging.info(f"Standard RMSE: {ret:.4f}")
    logging.info(f"Standard Matrix (K):\n{mtx}")
    logging.info(f"Standard Distortion (D):\n{dist}")

    save_calibration(args.output, ret, mtx, dist, "Standard Pinhole")

    # --- RATIONAL MODEL CALIBRATION ---
    logging.info("--- Rational Model Calibration (8 coeffs) ---")
    logging.info(f"Running cv2.calibrateCamera with CALIB_RATIONAL_MODEL...")
    # Rational model adds k4, k5, k6 to the distortion coefficients
    flags_rational = cv2.CALIB_RATIONAL_MODEL
    ret_rat, mtx_rat, dist_rat, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None, flags=flags_rational)
    logging.info("Rational calibration finished.")
    logging.info(f"Rational RMSE: {ret_rat:.4f}")
    logging.info(f"Rational Matrix (K):\n{mtx_rat}")
    logging.info(f"Rational Distortion (D):\n{dist_rat}")

    # Determine rational output filename
    base, ext = os.path.splitext(args.output)
    rat_output = f"{base}_rational{ext}"
    save_calibration(rat_output, ret_rat, mtx_rat, dist_rat, "Rational")

    # --- FISHEYE CALIBRATION ---
    logging.info("--- Fisheye Calibration ---")

    # Fisheye requires slightly different flags and carefully initialized arrays usually,
    # but often works with mostly defaults or specific flags.
    # We must ensure object points are (1, N, 3) and image points are (1, N, 2) which they should be.

    K_fish = np.zeros((3, 3))
    D_fish = np.zeros((4, 1))
    rvecs_fish = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
    tvecs_fish = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]

    flags_fish = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

    try:
        logging.info(f"Running cv2.fisheye.calibrate with {len(objpoints)} patterns... (this usually takes longer than standard)")
        ret_fish, K_fish, D_fish, rvecs_fish, tvecs_fish = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            img_size,
            K_fish,
            D_fish,
            rvecs_fish,
            tvecs_fish,
            flags_fish,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
        )
        logging.info("Fisheye calibration finished.")

        logging.info(f"Fisheye RMSE: {ret_fish:.4f}")
        logging.info(f"Fisheye Matrix (K):\n{K_fish}")
        logging.info(f"Fisheye Distortion (D):\n{D_fish}")

        # Determine fisheye output filename
        base, ext = os.path.splitext(args.output)
        fish_output = f"{base}_fisheye{ext}"
        save_calibration(fish_output, ret_fish, K_fish, D_fish, "Fisheye")

    except cv2.error as e:
        logging.error(f"Fisheye calibration failed: {e}")
        logging.warning("Skipping fisheye save used to standard calibration issues or data quality.")


def save_calibration(filename, rmse, mtx, dist, model_name):
    try:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Parameter", "Value"])
            writer.writerow(["Model", model_name])
            writer.writerow(["RMSE", rmse])

            # Intrinsics
            writer.writerow(["fx", mtx[0, 0]])
            writer.writerow(["fy", mtx[1, 1]])
            writer.writerow(["cx", mtx[0, 2]])
            writer.writerow(["cy", mtx[1, 2]])

            # Distortion coeffs
            dist_flat = dist.flatten()
            for i, d in enumerate(dist_flat):
                writer.writerow([f"dist_{i}", d])

        logging.info(f"Saved {model_name} intrinsics to {filename}")

    except IOError as e:
        logging.error(f"Error saving {filename}: {e}")


if __name__ == "__main__":
    main()
