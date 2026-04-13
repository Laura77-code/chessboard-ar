"""
Ned Flanders AR Chessboard Overlay (HW4 Robust Version)
----------------------------------
This script detects a chessboard in a video, estimates the camera pose,
and overlays an animated Ned Flanders GIF onto the board in perspective.

Preferred workflow:
1. Run calibration.py first with dedicated calibration videos.
2. Run this script to perform AR overlay using the saved calibration.
"""

import cv2 as cv
import numpy as np
import imageio
import os
import json
import argparse
import sys

# --- Configuration (Defaults) ---
CHESSBOARD_VIDEO = "example.mp4"
GIF_PATH = "nedFlanders.gif"
OUTPUT_VIDEO = "output.mp4"
CALIBRATION_FILE = "outputs/calibration/calibration_result.json"

# Chessboard settings
BOARD_PATTERN = (9, 6)  # Internal corners (width, height)
CELL_SIZE = 25.0         # Size of one square in mm

# AR Overlay settings
START_X, START_Y = 2, 1
SCALE_CELLS = 3.0
GIF_FRAME_DELAY = 3

# Sub-pixel refinement criteria
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def load_gif_frames(path):
    """
    Loads GIF frames and ensures they have an alpha channel for transparency.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"GIF file not found at {path}")
    
    print(f"Loading GIF from {path}...")
    gif = imageio.mimread(path)
    processed_frames = []
    
    for frame in gif:
        frame = np.array(frame)
        if frame.shape[2] == 3:
            frame = cv.cvtColor(frame, cv.COLOR_RGB2RGBA)
        else:
            frame = cv.cvtColor(frame, cv.COLOR_RGBA2BGRA)
        processed_frames.append(frame)
        
    print(f"Loaded {len(processed_frames)} frames.")
    return processed_frames

def load_and_scale_calibration(calib_path, target_size):
    """
    Loads calibration and scales K if resolution differs but aspect ratio is same.
    target_size is (width, height)
    """
    if not os.path.exists(calib_path):
        return None, None
        
    with open(calib_path, 'r') as f:
        data = json.load(f)
    
    K = np.array(data["camera_matrix"])
    dist = np.array(data["dist_coeff"])
    calib_w, calib_h = data["image_size"]
    target_w, target_h = target_size

    if (calib_w, calib_h) == (target_w, target_h):
        print(f"Calibration resolution matches video: {target_size}")
        return K, dist

    # Check aspect ratio
    calib_aspect = calib_w / calib_h
    target_aspect = target_w / target_h
    
    if abs(calib_aspect - target_aspect) > 0.01:
        print(f"ERROR: Aspect ratio mismatch! Calibration: {calib_aspect:.2f}, Video: {target_aspect:.2f}")
        print(f"Calibration size: {calib_w}x{calib_h}, Video size: {target_w}x{target_h}")
        sys.exit(1)

    # Scale K
    scale_x = target_w / calib_w
    scale_y = target_h / calib_h
    
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x # fx
    K_scaled[1, 1] *= scale_y # fy
    K_scaled[0, 2] *= scale_x # cx
    K_scaled[1, 2] *= scale_y # cy
    
    print(f"Resolution differs but aspect ratio is the same. Scaled K by ({scale_x:.2f}, {scale_y:.2f})")
    return K_scaled, dist

def get_robust_corners(gray, board_pattern):
    """
    Robustly finds chessboard corners using SB and classic fallback.
    """
    # Try FindChessboardCornersSB
    flags_sb = cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
    ret, corners = cv.findChessboardCornersSB(gray, board_pattern, flags_sb)
    
    if not ret:
        ret, corners = cv.findChessboardCorners(gray, board_pattern, None)
        if ret:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
    
    return ret, corners

def main():
    parser = argparse.ArgumentParser(description="Ned Flanders AR Chessboard Overlay")
    parser.add_argument("--input_video", default=CHESSBOARD_VIDEO, help="Input chessboard video")
    parser.add_argument("--gif", default=GIF_PATH, help="Animated GIF path")
    parser.add_argument("--calibration", default=CALIBRATION_FILE, help="Calibration JSON path")
    parser.add_argument("--output", default=OUTPUT_VIDEO, help="Output video path")
    
    args = parser.parse_args()

    # 1. Video processing setup (to get resolution first)
    cap = cv.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Could not open input video {args.input_video}")
        return

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)

    # 2. Calibration handling
    K, dist_coeffs = load_and_scale_calibration(args.calibration, (width, height))
    
    if K is None:
        print(f"\nERROR: Calibration file not found at {args.calibration}")
        print("Please run calibration first using calibration.py:")
        print(f"Example: python calibration.py --vids raw/vid1.mp4 raw/vid2.mp4")
        return

    # 3. Final setup
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(args.output, fourcc, fps, (width, height))

    try:
        gif_frames = load_gif_frames(args.gif)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 3D points for pose estimation
    objp = np.zeros((BOARD_PATTERN[0] * BOARD_PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_PATTERN[0], 0:BOARD_PATTERN[1]].T.reshape(-1, 2)
    objp *= CELL_SIZE

    # 4. Main processing loop
    frame_idx = 0
    last_rvec, last_tvec = None, None
    
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, corners = get_robust_corners(gray, BOARD_PATTERN)

        rvec, tvec = None, None
        if found:
            success, rvec, tvec = cv.solvePnP(objp, corners, K, dist_coeffs)
            if success:
                last_rvec, last_tvec = rvec, tvec
        else:
            rvec, tvec = last_rvec, last_tvec

        if rvec is not None and tvec is not None:
            gif_idx = (frame_idx // GIF_FRAME_DELAY) % len(gif_frames)
            overlay_img = gif_frames[gif_idx]
            oh, ow = overlay_img.shape[:2]
            aspect = ow / oh
            
            gif_w_mm = SCALE_CELLS * CELL_SIZE
            gif_h_mm = gif_w_mm / aspect
            
            model_pts = np.float32([
                [0, 0, -gif_h_mm], [0, 0, 0], [gif_w_mm, 0, -gif_h_mm], [gif_w_mm, 0, 0]
            ]) + np.float32([START_X * CELL_SIZE, START_Y * CELL_SIZE, 0])

            img_pts, _ = cv.projectPoints(model_pts, rvec, tvec, K, dist_coeffs)
            dst_pts = img_pts.reshape(-1, 2).astype(np.float32)
            src_pts = np.float32([[0, 0], [0, oh], [ow, 0], [ow, oh]])

            M = cv.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv.warpPerspective(overlay_img, M, (width, height), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

            overlay_bgr = warped[:, :, :3]
            alpha_mask = warped[:, :, 3] / 255.0
            for c in range(3):
                frame[:, :, c] = (frame[:, :, c] * (1.0 - alpha_mask) + overlay_bgr[:, :, c] * alpha_mask).astype(np.uint8)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0: print(f"Processed {frame_idx} frames...")

    print(f"Finished. Output saved to {args.output}")
    cap.release()
    out.release()

if __name__ == "__main__":
    main()
