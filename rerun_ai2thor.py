# Save this file as e.g., log_thor_to_rerun_matrix.py (NOT rerun.py)
import rerun as rr
import numpy as np
import json
import os
import math
import argparse
from PIL import Image
from scipy.spatial.transform import Rotation as R  # Import SciPy Rotation

from time import sleep

# Default base directory (assuming the script is run one level above the output folder)
DEFAULT_BASE_DIR = "thor_output"  # Or 'thor_output' depending on which you generated
DEFAULT_JSON_FILE = os.path.join(DEFAULT_BASE_DIR, "trajectory.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize AI2-THOR trajectory using Rerun."
    )
    parser.add_argument(
        "json_file",
        nargs="?",  # Make argument optional
        default=DEFAULT_JSON_FILE,
        help=f"Path to the trajectory JSON file (default: {DEFAULT_JSON_FILE})",
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help=f"Base directory containing rgb_images and depth_images folders (default: {DEFAULT_BASE_DIR})",
    )
    return parser.parse_args()


def log_trajectory(json_path, base_dir):
    """Loads trajectory data and logs it to Rerun using rotation matrices."""

    print(f"Loading trajectory from: {json_path}")
    print(f"Using base directory for images: {os.path.abspath(base_dir)}")

    with open(json_path, "r") as f:
        trajectory_data = json.load(f)

    # --- Initialize Rerun ---
    app_id = "ai2thor_new"
    # The 'timeless' argument for init might also be deprecated depending on version,
    rr.init(
        app_id,
        spawn=True,
    )

    # --- Log Coordinate System ---
    # Log timelessly using the rr.log function's parameter
    # AI2thor is RUF (Right-Up-Forward)
    rr.log("world", rr.ViewCoordinates.RUF)

    print(f"Logging {len(trajectory_data)} steps to Rerun...")

    for i, step_data in enumerate(trajectory_data):
        step = step_data.get("step", i)
        # --- Process Pose and Images ---
        pos = step_data["position"]
        rot = step_data["rotation"]
        camera_horizon_deg = np.float128(step_data.get("cameraHorizon", 0.0))
        agent_roll_deg = np.float128(rot.get("x", 0.0))
        agent_yaw_deg = np.float128(rot.get("y", 0.0))
        agent_pitch_deg = np.float128(rot.get("z", 0.0))
        # print(f"Step {step}: Pos({pos}), Rot({rot}), Horizon({camera_horizon_deg})")
        total_pitch_deg = camera_horizon_deg + agent_pitch_deg

        # --- Coordinate System Conversion (Position) ---
        x, y, z_thor = (
            np.float128(pos.get("x", 0.0)),
            np.float128(pos.get("y", 0.0)),
            np.float128(pos.get("z", 0.0)),
        )
        print(x, y, z_thor)
        translation = [x, y, z_thor]  # Negate Z for Rerun

        # --- Rotation Conversion (Euler to Matrix using SciPy) ---
        # yaw_rad = math.radians(agent_yaw_deg)
        # pitch_rad = math.radians(
        # total_pitch_deg
        # )  # Positive horizon = look down = pitch around camera X
        # roll_rad = math.radians(agent_roll_deg)
        # final_yaw_rad = yaw_rad + math.pi  # Add 180 deg to Yaw for Z flip

        # Use SciPy to convert from Euler angles to a rotation matrix.
        # We use the extrinsic 'YXZ' sequence which corresponds to Yaw, Pitch, Roll
        # applied sequentially to the world axes. This matches the derivation
        # for aligning AI2-THOR's frame (+Z fwd) to Rerun's camera frame (-Z fwd).
        # scipy_rot = R.from_euler(
        # "YXZ",  # Extrinsic Euler sequence: Yaw(Y), Pitch(X), Roll(Z)
        # [final_yaw_rad, pitch_rad, roll_rad],
        # degrees=False,  # Input angles are in radians
        # )
        # Get the 3x3 rotation matrix\

        scipy_rot = R.from_euler(
            "YXZ", [agent_yaw_deg, total_pitch_deg, agent_roll_deg], degrees=True
        )
        rotation_matrix_np = scipy_rot.as_matrix()

        # *** Log Transform using Mat3x3 ***
        transform = rr.Transform3D(
            translation=translation,
            mat3x3=rotation_matrix_np,  # Pass the matrix here
        )
        # print(translation, rotation_matrix_np)

        rr.log(f"world/camera/{i}", transform)

        # rr.log(f"world/point/{i}", rr.Points3D(positions=[translation], radii=[0.01]))

        rgb_path_rel = step_data.get("rgb_image_path")
        depth_path_rel = step_data.get("depth_npy_path")

        assert os.path.exists(
            os.path.join(base_dir, rgb_path_rel)
        ), f"RGB image path does not exist: {rgb_path_rel}"

        rgb_full_path = os.path.join(base_dir, rgb_path_rel)
        depth_full_path = os.path.join(base_dir, depth_path_rel)

        assert os.path.exists(
            os.path.join(base_dir, depth_path_rel)
        ), f"Depth image path does not exist: {depth_path_rel}"

        if not os.path.exists(rgb_full_path):
            print(f"Error: RGB image path does not exist: {rgb_full_path}")
            continue

        with Image.open(rgb_full_path) as img_pil:
            rgb_image = np.array(img_pil)

        depth_image = np.load(depth_full_path)

        if rgb_image is not None and rgb_image.ndim == 3:
            print(f"Step {step}: RGB image shape: {rgb_image.shape}")
            h, w, _ = rgb_image.shape
            hfov_rad = math.radians(90.0)
            # fx = fy = w / (2.0 * math.tan(hfov_rad / 2.0))
            fx = fy = w  # for this
            cx = w / 2.0
            cy = h / 2.0
            camera_pinhole = rr.Pinhole(
                image_from_camera=[
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                resolution=(w, h),
                camera_xyz=rr.ViewCoordinates.RUF,
                # camera_xyz=rr.ViewCoordinates.LUF,
            )

            rr.log(f"world/camera/{i}", camera_pinhole)

            rr.log(
                f"world/camera/{i}/rgb",
                rr.Image(rgb_image),  # .compress(jpeg_quality=20)
            )
            # rr.log(
            #     f"world/camera/{i}/depth",
            #     rr.DepthImage(depth_image),  # .compress(jpeg_quality=20)
            # )
            print(f"Logged camera intrinsics: {camera_pinhole}")
            print(
                f"Logged camera intrinsics (w={w}, h={h}, f=({fx:.2f},{fy:.2f}), c=({cx:.1f},{cy:.1f}))"
            )

    print("Finished logging.")


# if __name__ == "__main__":
args = parse_args()

# --- Input Validation ---
if not os.path.isfile(args.json_file):
    print(f"Error: JSON file does not exist or is not a file: '{args.json_file}'")
    exit(1)
if not os.path.isdir(args.base_dir):
    print(
        f"Error: Base directory does not exist or is not a directory: '{args.base_dir}'"
    )
    exit(1)
rgb_dir_check = os.path.join(args.base_dir, "rgb_images")
depth_dir_check = os.path.join(args.base_dir, "depth_images")
if not os.path.isdir(rgb_dir_check):
    print(
        f"Warning: Expected RGB directory '{rgb_dir_check}' not found within base directory."
    )
if not os.path.isdir(depth_dir_check):
    print(
        f"Warning: Expected Depth directory '{depth_dir_check}' not found within base directory."
    )

log_trajectory(args.json_file, args.base_dir)
