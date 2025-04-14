import ai2thor.controller
import os
import json
from PIL import Image
import numpy as np
import time
from random import choice

# --- Configuration ---
SCENE_NAME = "FloorPlan211"  # Example scene, choose any valid scene name
OUTPUT_DIR = "thor_output"  # Directory to save all outputs
RGB_DIR = os.path.join(OUTPUT_DIR, "rgb_images")
DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth_images")
TRAJ_FILE = os.path.join(OUTPUT_DIR, "trajectory.json")

# Agent movement parameters
GRID_SIZE = 0.25  # Movement step size in meters
ROTATION_DEGREES = 20  # Rotation step size in degrees
STEPS = 40
# --- Setup Output Directories ---
os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)
print(f"Output will be saved in: {os.path.abspath(OUTPUT_DIR)}")

# --- Initialize AI2-THOR Controller ---
print(f"Initializing AI2-THOR Controller for scene: {SCENE_NAME}...")

try:
    # controller = ai2thor.controller.Controller(
    #     scene=SCENE_NAME,
    #     gridSize=GRID_SIZE,
    #     rotateStepDegrees=ROTATION_DEGREES,
    #     renderDepthImage=True,  # Crucial for getting depth frames
    #     renderInstanceSegmentation=False, # Optional, disable if not needed
    #     width=640, # Optional: Set resolution
    #     height=480, # Optional: Set resolution
    #     # Add platform specific options if needed, e.g., for Linux:
    #     # platform=ai2thor.platform.CloudRendering
    #     # local_executable_path='/path/to/your/ai2thor/executable' # If needed
    # )
    controller = ai2thor.controller.Controller(
        agentMode="default",
        visibilityDistance=30,
        scene=SCENE_NAME,
        # step sizes
        gridSize=GRID_SIZE,
        snapToGrid=False,
        rotateStepDegrees=ROTATION_DEGREES,
        # image modalities
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        # camera properties
        width=300,
        height=300,
        fieldOfView=90,
    )
    print("Controller initialized successfully.")
except Exception as e:
    print(f"Error initializing controller: {e}")
    print("Please ensure AI2-THOR is installed and accessible.")
    exit()


# print all possible actions
print("Available actions:")
# --- Start the Controller ---
# --- Define Agent Actions ---
# Sequence of actions the agent will perform
actions = [
    "MoveAhead",
    "MoveAhead",
    # "RotateRight",
    "MoveAhead",
    "LookDown",
    "MoveAhead",
    "LookUp",
    # "RotateLeft",
    # "RotateLeft",
    # "RotateLeft",
    # "RotateLeft",
    # "RotateLeft",
    # "RotateLeft",
    # "RotateLeft",
    # "RotateLeft",
    # "RotateLeft",
    # "RotateLeft",
    # "RotateLeft",
    # "RotateLeft",
    "MoveBack",
    "RotateRight",
    "RotateRight",
    "RotateRight",
    "RotateRight",
    "RotateRight",
    "RotateRight",
    "RotateRight",
    "RotateRight",
    "RotateRight",
    # "RotateRight",
    # "RotateRight",
    # "RotateRight",
    # "RotateRight",
    # "RotateRight",
    # "RotateRight",
    # "RotateRight",
    "MoveLeft",  # Example of strafing
    "MoveRight",  # Example of strafing
]

# --- Data Collection ---
trajectory_data = []
step_count = 0


# Function to save images and record state
def process_step(event, action_taken, step_num):
    """Saves images and records agent state from an event."""
    # if not event.metadata["lastActionSuccess"]:
    #     print(
    #         f"  Action '{action_taken}' failed at step {step_num}. Skipping data save for this step."
    #     )
    #     # Record failure in trajectory
    #     agent_meta = event.metadata["agent"]
    #     traj_point = {
    #         "step": step_num,
    #         "action_taken": action_taken,
    #         "action_success": False,
    #         "position": agent_meta["position"],
    #         "rotation": agent_meta["rotation"],
    #         "cameraHorizon": agent_meta["cameraHorizon"],
    #         "rgb_image_path": None,
    #         "depth_image_path": None,
    #         "depth_npy_path": None,  # Added for raw depth
    #     }
    #     return traj_point  # Return data even on failure

    # --- Save RGB Image ---
    rgb_frame = event.frame  # This is a numpy array (H, W, C)
    rgb_img = Image.fromarray(rgb_frame)
    rgb_filename = f"rgb_step_{step_num:04d}.png"
    rgb_filepath = os.path.join(RGB_DIR, rgb_filename)
    rgb_img.save(rgb_filepath)
    # print(f"  Saved RGB image: {rgb_filepath}")

    # --- Save Depth Image ---
    # depth_frame is in meters (float32 numpy array)
    depth_frame_meters = event.depth_frame

    # Option 1: Save raw depth data as .npy (Recommended for accuracy)
    depth_npy_filename = f"depth_step_{step_num:04d}.npy"
    depth_npy_filepath = os.path.join(DEPTH_DIR, depth_npy_filename)
    np.save(depth_npy_filepath, depth_frame_meters)
    # print(f"  Saved raw depth data: {depth_npy_filepath}")

    # Option 2: Save visualized depth as PNG (Normalized, loses precision)
    # Normalize depth for visualization (e.g., scale to 0-255)
    # Handle potential infinite values or very large distances if necessary
    valid_depth = depth_frame_meters[np.isfinite(depth_frame_meters)]
    if len(valid_depth) > 0:
        depth_min, depth_max = valid_depth.min(), valid_depth.max()
        if depth_max > depth_min:  # Avoid division by zero
            # Simple normalization, adjust clipping/scaling as needed
            depth_normalized = (depth_frame_meters - depth_min) / (
                depth_max - depth_min
            )
            depth_visual = (np.clip(depth_normalized, 0, 1) * 255).astype(np.uint8)
        else:  # Handle case where all valid depths are the same
            depth_visual = np.zeros_like(depth_frame_meters, dtype=np.uint8)
    else:  # Handle case with no valid depth values
        depth_visual = np.zeros_like(depth_frame_meters, dtype=np.uint8)

    depth_img = Image.fromarray(depth_visual)
    depth_filename = f"depth_viz_step_{step_num:04d}.png"  # Note 'viz' in name
    depth_filepath = os.path.join(DEPTH_DIR, depth_filename)
    depth_img.save(depth_filepath)
    # print(f"  Saved visualized depth image: {depth_filepath}")

    # --- Record Trajectory Point ---
    agent_meta = event.metadata["agent"]
    traj_point = {
        "step": step_num,
        "action_taken": action_taken,
        "action_success": True,
        "position": agent_meta["position"],
        "rotation": agent_meta["rotation"],
        "cameraHorizon": agent_meta["cameraHorizon"],
        "rgb_image_path": os.path.relpath(rgb_filepath, OUTPUT_DIR),  # Relative path
        "depth_image_path": os.path.relpath(
            depth_filepath, OUTPUT_DIR
        ),  # Relative path for viz
        "depth_npy_path": os.path.relpath(
            depth_npy_filepath, OUTPUT_DIR
        ),  # Relative path for raw
    }
    return traj_point


# --- Process Initial State (Before Any Action) ---
print("\nProcessing initial state (Step 0)...")
initial_event = controller.last_event
initial_traj_point = process_step(initial_event, "InitialState", step_count)
if (
    initial_traj_point
):  # Check if processing was successful (it should be for initial state)
    trajectory_data.append(initial_traj_point)
    print(f"Step {step_count}: Initial State recorded.")


# --- Execute Actions and Save Data ---
print("\nExecuting action sequence...")
for action_step in range(0, STEPS):
    action = choice(actions)  # Randomly choose an action from the list
    step_count += 1
    print(f"Step {step_count}: Executing action '{action}'...")
    event = controller.step(action=action)

    # Process the result of the action
    traj_point = process_step(event, action, step_count)
    if traj_point:  # Always append, even failures are recorded
        trajectory_data.append(traj_point)

    # Optional: Add a small delay between steps if needed
    # time.sleep(0.1)

# --- Save Trajectory Data ---
print("\nSaving trajectory data...")
try:
    with open(TRAJ_FILE, "w") as f:
        json.dump(trajectory_data, f, indent=4)
    print(f"Trajectory saved successfully to: {TRAJ_FILE}")
except Exception as e:
    print(f"Error saving trajectory file: {e}")

# --- Stop the Controller ---
print("\nStopping the AI2-THOR controller...")
try:
    controller.stop()
    print("Controller stopped.")
except Exception as e:
    print(f"Error stopping controller: {e}")


# save a video of the images
import cmd
from glob import glob


def create_video_from_images(image_folder, output_video_path, fps=30):
    """Creates a video from a sequence of images in a folder."""
    # Get all image files in the folder
    image_files = sorted(glob(os.path.join(image_folder, "*.png")))
    if not image_files:
        print(f"No images found in {image_folder}. Cannot create video.")
        return
    cmd = f"ffmpeg -framerate {fps} -i {image_folder}/rgb_step_%04d.png -c:v libx264 -pix_fmt yuv420p {output_video_path}"
    os.system(cmd)
    print(f"Video created successfully: {output_video_path}")


# Create video from RGB images
video_output_path = os.path.join(OUTPUT_DIR, "rgb_video.mp4")
create_video_from_images(RGB_DIR, video_output_path, fps=30)
# Create video from depth images
# depth_video_output_path = os.path.join(OUTPUT_DIR, "depth_video.mp4")
# create_video_from_images(DEPTH_DIR, depth_video_output_path, fps=30)

print(f"RGB video saved to: {video_output_path}")
# print(f"Depth video saved to: {depth_video_output_path}")
# --- Final Output ---
print("\nData collection complete.")
print(f"RGB images saved to: {RGB_DIR}")
print(f"Depth images saved to: {DEPTH_DIR}")
print(f"Trajectory data saved to: {TRAJ_FILE}")
print(f"Video saved to: {video_output_path}")
# print(f"Depth video saved to: {depth_video_output_path}")
print("All data has been collected and saved.")
print("\nSimulation finished.")
