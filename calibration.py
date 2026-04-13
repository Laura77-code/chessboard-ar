import cv2 as cv
import numpy as np
import json
import os
import argparse
import csv
from datetime import datetime

class RobustCalibrator:
    def __init__(self, board_pattern=(9, 6), cell_size=25.0, blur_threshold=30.0, 
                 duplicate_threshold=15.0, max_frames=60):
        self.board_pattern = board_pattern
        self.cell_size = cell_size
        self.blur_threshold = blur_threshold
        self.duplicate_threshold = duplicate_threshold
        self.max_frames = max_frames
        
        self.obj_points = [] # 3D points
        self.img_points = [] # 2D points
        self.frame_metadata = [] # To store per-frame diagnostics
        self.accepted_corners = []
        self.accepted_centers = [] # For spatial diversity
        
        # Prepare object points once
        self.objp = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2)
        self.objp *= cell_size

    def get_blur_score(self, gray):
        return cv.Laplacian(gray, cv.CV_64F).var()

    def is_redundant(self, new_corners):
        """
        Checks if the new detection is too similar to existing ones
        based on average corner distance.
        """
        if not self.accepted_corners:
            return False
        
        for old_corners in self.accepted_corners:
            dist = np.mean(np.linalg.norm(old_corners - new_corners, axis=2))
            if dist < self.duplicate_threshold:
                return True
        return False

    def find_corners(self, gray):
        # Try FindChessboardCornersSB if available (OpenCV 4.x)
        # CALIB_CB_EXHAUSTIVE is powerful but slow, good for calibration
        flags_sb = cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
        ret, corners = cv.findChessboardCornersSB(gray, self.board_pattern, flags_sb)
        
        if not ret:
            # Fallback to classic method
            ret, corners = cv.findChessboardCorners(gray, self.board_pattern, None)
            if ret:
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return ret, corners

    def calibrate(self, video_paths, frame_step=10):
        print(f"Starting robust calibration on {len(video_paths)} videos...")
        image_size = None
        
        stats = {
            "total_frames_checked": 0, 
            "rejected_blur": 0, 
            "rejected_detection": 0, 
            "rejected_redundant": 0, 
            "accepted": 0
        }

        for v_path in video_paths:
            if not os.path.exists(v_path):
                print(f"Warning: Video file {v_path} not found. Skipping.")
                continue

            cap = cv.VideoCapture(v_path)
            if not cap.isOpened():
                print(f"Warning: Could not open {v_path}")
                continue
                
            f_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if f_idx % frame_step != 0:
                    f_idx += 1
                    continue
                
                stats["total_frames_checked"] += 1
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                if image_size is None: image_size = gray.shape[::-1]
                
                # 1. Blur filter (Laplacian variance)
                blur_score = self.get_blur_score(gray)
                if blur_score < self.blur_threshold:
                    stats["rejected_blur"] += 1
                    f_idx += 1
                    continue
                
                # 2. Chessboard detection
                found, corners = self.find_corners(gray)
                if not found:
                    stats["rejected_detection"] += 1
                    f_idx += 1
                    continue
                
                # 3. Redundancy filter (Spatial diversity / duplicate rejection)
                if self.is_redundant(corners):
                    stats["rejected_redundant"] += 1
                    f_idx += 1
                    continue
                
                # Accept frame
                self.img_points.append(corners)
                self.obj_points.append(self.objp)
                self.accepted_corners.append(corners)
                
                # Calculate center for diversity tracking
                center = np.mean(corners, axis=0)
                self.accepted_centers.append(center)
                
                self.frame_metadata.append({
                    "video": os.path.basename(v_path), 
                    "frame": f_idx, 
                    "blur": f"{blur_score:.2f}"
                })
                stats["accepted"] += 1
                
                if len(self.img_points) >= self.max_frames:
                    print(f"Reached maximum frame limit ({self.max_frames}) for calibration.")
                    break
                
                f_idx += 1
            cap.release()
            if len(self.img_points) >= self.max_frames: break

        print(f"Calibration data collection finished.")
        print(f"Summary: {stats}")
        
        if len(self.img_points) < 15:
            raise ValueError(f"Insufficient frames for robust calibration. Found only {len(self.img_points)} frames. Try reducing blur/duplicate thresholds or adding more videos.")

        # Run OpenCV Calibration
        print("Computing camera matrix and distortion coefficients...")
        ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
            self.obj_points, self.img_points, image_size, None, None
        )
        
        # Compute reprojection errors
        total_error = 0
        for i in range(len(self.obj_points)):
            imgpoints2, _ = cv.projectPoints(self.obj_points[i], rvecs[i], tvecs[i], K, dist)
            error = cv.norm(self.img_points[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            total_error += error
            self.frame_metadata[i]["reprojection_error"] = f"{error:.6f}"

        mean_error = total_error / len(self.obj_points)
        print(f"Calibration completed. RMS: {ret:.4f}, Mean Reprojection Error: {mean_error:.4f}")
        
        return {
            "ret": ret,
            "K": K,
            "dist": dist,
            "mean_error": mean_error,
            "image_size": image_size,
            "num_frames": len(self.img_points),
            "metadata": self.frame_metadata
        }

    def save_results(self, result, output_dir="outputs/calibration"):
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. JSON result (Main output for pose estimation)
        json_data = {
            "camera_matrix": result["K"].tolist(),
            "dist_coeff": result["dist"].tolist(),
            "rms": result["ret"],
            "mean_reprojection_error": result["mean_error"],
            "image_size": result["image_size"],
            "num_frames": result["num_frames"],
            "date_generated": datetime.now().isoformat()
        }
        json_path = os.path.join(output_dir, "calibration_result.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        print(f"Saved: {json_path}")
            
        # 2. Per-frame errors CSV for diagnostics
        csv_path = os.path.join(output_dir, "per_frame_errors.csv")
        with open(csv_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["video", "frame", "blur", "reprojection_error"])
            writer.writeheader()
            writer.writerows(result["metadata"])
        print(f"Saved: {csv_path}")
            
        # 3. Summary Report MD
        report_path = os.path.join(output_dir, "calibration_report.md")
        with open(report_path, "w") as f:
            f.write("# Robust Camera Calibration Report\n\n")
            f.write(f"- **Status**: SUCCESS\n")
            f.write(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Frames Used**: {result['num_frames']}\n")
            f.write(f"- **Original Image Size**: {result['image_size']} (W, H)\n")
            f.write(f"- **RMS Error**: {result['ret']:.6f}\n")
            f.write(f"- **Mean Reprojection Error**: {result['mean_error']:.6f}\n\n")
            f.write("## Intrinsic Matrix (K)\n")
            f.write(f"```\n{result['K']}\n```\n")
            f.write("## Distortion Coefficients\n")
            f.write(f"```\n{result['dist']}\n```\n")
        print(f"Saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Robust Camera Calibration for HW4")
    parser.add_argument("--vids", nargs="+", help="Paths to calibration videos")
    parser.add_argument("--pattern", nargs=2, type=int, default=[9, 6], help="Internal corners (W H)")
    parser.add_argument("--size", type=float, default=25.0, help="Square size in mm")
    parser.add_argument("--step", type=int, default=5, help="Frame step for sampling")
    parser.add_argument("--blur", type=float, default=30.0, help="Minimum Laplacian variance score")
    parser.add_argument("--max_frames", type=int, default=60, help="Maximum frames to use")
    parser.add_argument("--output_dir", default="outputs/calibration", help="Where to save results")
    
    args = parser.parse_args()
    
    if not args.vids:
        print("Error: No calibration videos provided. Use --vids path/to/video1 path/to/video2 ...")
        return

    calibrator = RobustCalibrator(
        board_pattern=tuple(args.pattern),
        cell_size=args.size,
        blur_threshold=args.blur,
        max_frames=args.max_frames
    )
    
    try:
        result = calibrator.calibrate(args.vids, frame_step=args.step)
        calibrator.save_results(result, output_dir=args.output_dir)
        print("\nRobust calibration complete. You can now run the AR pipeline using these results.")
    except Exception as e:
        print(f"\nCalibration failed: {e}")

if __name__ == "__main__":
    main()
