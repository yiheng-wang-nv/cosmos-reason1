#!/usr/bin/env python3
"""
SO101 Real Robot Evaluation Script for GR00T Policy

Based on eval_gr00t_so100.py but adapted for SO101 robot.
This script connects to a GR00T inference server and runs policy evaluation on SO101.

Usage:
python so101_robot_utils.py \
    --host 127.0.0.1 \
    --port 5555 \
    --port_follower /dev/ttyACM2 \
    --task_description "Grip a straight scissor and put it in the box." \
    --actions_to_execute 30 \
    --monitor_cameras

python so101_robot_utils.py \
    --host 127.0.0.1 \
    --port 5555 \
    --port_follower /dev/ttyACM2 \
    --task_description "Grip a tweezer and put it in the box." \
    --actions_to_execute 30 \
    --monitor_cameras
"""

import argparse
import os
import time
from contextlib import contextmanager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.feetech import TorqueMode
from lerobot.common.robot_devices.robots.configs import So101RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError
from tqdm import tqdm

# Import GR00T client
from gr00t.eval.service import ExternalRobotInferenceClient


class SO101Robot:
    """SO101 Robot controller for GR00T policy evaluation."""
    
    def __init__(self, port_follower: str = "/dev/ttyACM2", calibrate: bool = False, 
                 enable_camera: bool = True):
        self.config = So101RobotConfig()
        self.calibrate = calibrate
        self.enable_camera = enable_camera
        self.cam_idx = (1, 0)
        self.port_follower = port_follower
        
        # Configure robot
        if not enable_camera:
            self.config.cameras = {}
        else:
            self.config.cameras = {
                "wrist": OpenCVCameraConfig(
                    camera_index=self.cam_idx[0],
                    fps=30,
                    width=640,
                    height=480,
                ),
                "room": OpenCVCameraConfig(
                    camera_index=self.cam_idx[1],
                    fps=30,
                    width=640,
                    height=480,
                )
            }
        
        # Inference mode: no leader arms needed
        self.config.leader_arms = {}
        
        # Set follower arm port
        self.config.follower_arms["main"].port = port_follower
        
        # Remove calibration folder if requested
        if self.calibrate:
            import shutil
            calibration_folder = os.path.join(os.getcwd(), ".cache", "calibration", "so101")
            print(f"========> Deleting calibration_folder: {calibration_folder}")
            if os.path.exists(calibration_folder):
                shutil.rmtree(calibration_folder)
        
        # Create the robot
        self.robot = make_robot_from_config(self.config)
        self.motor_bus = self.robot.follower_arms["main"]

    @contextmanager
    def activate(self):
        """Context manager for robot activation."""
        try:
            self.connect()
            self.move_to_initial_pose()
            yield
        finally:
            self.disconnect()

    def connect(self):
        """Connect to SO101 robot."""
        if self.robot.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "SO101Robot is already connected. Do not run `robot.connect()` twice."
            )

        # Connect the motor bus
        self.motor_bus.connect()

        # Disable torque for calibration
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

        # Run calibration
        self.robot.activate_calibration()

        # Set robot preset
        self.set_so101_robot_preset()

        # Enable torque
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)
        print("SO101 present position:", self.motor_bus.read("Present_Position"))
        self.robot.is_connected = True

        # Connect camera
        self.camera_wrist = self.robot.cameras["wrist"] if self.enable_camera else None
        self.camera_room = self.robot.cameras["room"] if self.enable_camera else None
        if self.camera_wrist is not None:
            self.camera_wrist.connect()
        if self.camera_room is not None:
            self.camera_room.connect()
        
        print("================> SO101 Robot is fully connected =================")

    def set_so101_robot_preset(self):
        """Set SO101-specific motor configurations."""
        # Mode=0 for Position Control
        self.motor_bus.write("Mode", 0)
        # Set P_Coefficient to lower value to avoid shakiness
        self.motor_bus.write("P_Coefficient", 16)
        # Set I_Coefficient and D_Coefficient
        self.motor_bus.write("I_Coefficient", 0)
        self.motor_bus.write("D_Coefficient", 32)
        # Close the write lock
        self.motor_bus.write("Lock", 0)
        # Set Maximum_Acceleration for faster response
        self.motor_bus.write("Maximum_Acceleration", 50)
        self.motor_bus.write("Acceleration", 50)

    def move_to_initial_pose(self):
        """Move robot to initial pose."""
        print("-------------------------------- Moving to initial pose")
        # SO101 initial pose (adjust these values as needed)
        initial_state = torch.tensor([8, 196, 180, 74, 95, 0], dtype=torch.float32)
        self.robot.send_action(initial_state)
        time.sleep(0.5)

    def go_home(self):
        """Move robot to home pose."""
        print("-------------------------------- Moving to home pose")
        # SO101 home pose (adjust these values as needed)
        home_state = torch.tensor([8, 196, 180, 74, 95, 0], dtype=torch.float32)
        self.set_target_state(home_state)
        time.sleep(0.5)

    def get_observation(self):
        """Get robot observation."""
        return self.robot.capture_observation()

    def get_current_state(self):
        """Get current robot state as numpy array."""
        return self.get_observation()["observation.state"].data.numpy()

    def get_current_img(self):
        """Get current camera image as RGB numpy array."""
        if not self.enable_camera:
            return np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640, 3), dtype=np.uint8)
        
        img = self.get_observation()["observation.images.wrist"].data.numpy()
        img_room = self.get_observation()["observation.images.room"].data.numpy()
        # Convert to HWC format if needed
        if len(img.shape) == 3 and img.shape[0] == 3:  # CHW format
            img = img.transpose(1, 2, 0)  # Convert to HWC
        if len(img_room.shape) == 3 and img_room.shape[0] == 3:  # CHW format
            img_room = img_room.transpose(1, 2, 0)  # Convert to HWC
        # Ensure uint8
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if img_room.dtype != np.uint8:
            img_room = (img_room * 255).astype(np.uint8)
        return img, img_room

    def get_camera_image(self, camera_name: str) -> np.ndarray:
        """Get current image from a specified camera as an RGB numpy array."""
        if not self.enable_camera:
            print(f"Cameras are disabled. Returning a black image for {camera_name}.")
            # Assuming default camera dimensions if not available otherwise
            # You might want to get this from config if available
            return np.zeros((480, 640, 3), dtype=np.uint8)

        if camera_name not in self.config.cameras:
            raise ValueError(f"Camera '{camera_name}' not found in robot configuration. Available cameras: {list(self.config.cameras.keys())}")

        obs = self.get_observation()
        image_key = f"observation.images.{camera_name}"
        
        if image_key not in obs:
            raise KeyError(f"Image key '{image_key}' not found in observation. Available keys: {list(obs.keys())}")

        img_data = obs[image_key].data.numpy()

        # Convert to HWC format if needed
        if len(img_data.shape) == 3 and img_data.shape[0] == 3:  # CHW format
            img_data = img_data.transpose(1, 2, 0)  # Convert to HWC
        
        # Ensure uint8
        if img_data.dtype != np.uint8:
            # Check if values are in [0, 1] range before scaling
            if img_data.max() <= 1.0 and img_data.min() >= 0.0:
                img_data = (img_data * 255).astype(np.uint8)
            else:
                # If not in [0,1], try to cast directly, or handle as an error
                try:
                    img_data = img_data.astype(np.uint8)
                except ValueError as e:
                    print(f"Warning: Could not convert image data for {camera_name} to uint8 directly. Values might be out of [0, 255] range or not in [0,1] for scaling. Error: {e}")
                    # Return a black image or raise an error
                    return np.zeros((img_data.shape[1], img_data.shape[2], 3) if len(img_data.shape) == 3 and img_data.shape[0] == 3 else (480,640,3), dtype=np.uint8)


        return img_data

    def set_target_state(self, target_state: torch.Tensor):
        """Send target state to robot."""
        self.robot.send_action(target_state)

    def enable(self):
        """Enable motor torque."""
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)

    def disable(self):
        """Disable motor torque."""
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

    def disconnect(self):
        """Disconnect robot."""
        self.disable()
        if self.robot.is_connected:
            self.robot.disconnect()
            self.robot.is_connected = False
        print("================> SO101 Robot disconnected")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'robot') and self.robot.is_connected:
            self.disconnect()


class Gr00tSO101InferenceClient:
    """GR00T inference client for SO101."""
    
    def __init__(self, host: str = "localhost", port: int = 5555, 
                 language_instruction: str = "Pick up the object and place it in the box."):
        self.language_instruction = language_instruction
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        print(f"Connected to GR00T server at {host}:{port}")
        print(f"Task: {language_instruction}")

    def get_action(self, img: np.ndarray, img_room: np.ndarray, state: np.ndarray) -> dict:
        """Get action from GR00T policy."""
        # Format observation for SO101 wrist config
        obs_dict = {
            "video.wrist": img[np.newaxis, :, :, :],  # Add batch dimension
            "video.room": img_room[np.newaxis, :, :, :],  # Add batch dimension
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),  # 5 arm joints
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),   # 1 gripper joint
            "annotation.human.task_description": [self.language_instruction],
        }
        return self.policy.get_action(obs_dict)

    def set_language_instruction(self, instruction: str):
        """Update task instruction."""
        self.language_instruction = instruction
        print(f"Task updated: {instruction}")


def setup_arg_parser():
    """Sets up and returns the argument parser."""
    parser = argparse.ArgumentParser(description="SO101 GR00T Policy Evaluation")
    
    # GR00T server connection
    parser.add_argument("--host", type=str, default="localhost", help="GR00T server host")
    parser.add_argument("--port", type=int, default=5555, help="GR00T server port")
    
    # Robot configuration
    parser.add_argument("--port_follower", type=str, default="/dev/ttyACM1", help="SO101 serial port")
    
    # Task configuration
    parser.add_argument("--task_description", type=str, 
                       default="Grip a straight scissor and put it in the box.",
                       help="Task description for the policy")
    
    # Execution parameters
    parser.add_argument("--actions_to_execute", type=int, default=300, help="Number of action steps")
    parser.add_argument("--action_horizon", type=int, default=12, help="Actions to execute from chunk")
    parser.add_argument("--calibrate", action="store_true", help="Run robot calibration")
    parser.add_argument("--record_images", action="store_true", help="Save images during execution")
    parser.add_argument("--create_videos", action="store_true", help="Create videos from saved images")
    parser.add_argument("--video_fps", type=int, default=15, help="FPS for created videos")
    parser.add_argument("--output_dir", type=str, default="eval_so101_images", help="Output directory for images")
    parser.add_argument("--monitor_cameras", action="store_true", help="Display live camera feeds in windows")
    return parser


def setup_image_recording(args):
    """Sets up the directory for image recording if requested."""
    if args.record_images:
        os.makedirs(args.output_dir, exist_ok=True)
        # Clear existing images
        for file in os.listdir(args.output_dir):
            if file.endswith(('.jpg', '.png', '.mp4')):
                os.remove(os.path.join(args.output_dir, file))
        print(f"Recording images to: {args.output_dir}")


def run_camera_preview(robot: SO101Robot) -> bool:
    """
    Runs a camera preview loop.
    Returns True if the user wants to start, False if they want to quit.
    """
    print("\nCamera preview is active.")
    print("Check the camera windows. Press 's' to start policy execution, or 'q' to quit.")
    while True:
        img_preview, img_room_preview = robot.get_current_img()
        
        # Convert to BGR for OpenCV display
        img_wrist_bgr = cv2.cvtColor(img_preview, cv2.COLOR_RGB2BGR)
        img_room_bgr = cv2.cvtColor(img_room_preview, cv2.COLOR_RGB2BGR)

        cv2.imshow("Wrist Camera", img_wrist_bgr)
        cv2.imshow("Room Camera", img_room_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("Starting policy execution...")
            return True
        elif key == ord('q'):
            print("Quitting application from preview.")
            return False


def execute_policy_loop(robot: SO101Robot, client: Gr00tSO101InferenceClient, args) -> int:
    """
    Contains the main policy execution loop.
    Returns the final count of images recorded.
    """
    image_count = 0
    MODALITY_KEYS = ["single_arm", "gripper"]
    execution_stopped = False

    for i in tqdm(range(args.actions_to_execute), desc="Executing actions"):
        # Get current observation for the policy
        img, img_room = robot.get_current_img()
        state = robot.get_current_state()
        
        # Get action from policy
        action_start_time = time.time()
        action = client.get_action(img, img_room, state)
        
        # Execute action chunk
        execution_start_time = time.time()
        for j in range(args.action_horizon):
            # Concatenate action components
            concat_action = np.concatenate([
                np.atleast_1d(action[f"action.{key}"][j]) 
                for key in MODALITY_KEYS
            ], axis=0)
            
            assert concat_action.shape == (6,), f"Expected (6,) but got {concat_action.shape}"
            
            # Send to robot
            robot.set_target_state(torch.from_numpy(concat_action))
            time.sleep(0.05)  # Small delay between actions
            
            # Get a fresh frame for monitoring and saving
            live_img, live_img_room = robot.get_current_img()
            
            # Save images if recording
            if args.record_images:
                # Resize and save wrist camera image
                img_wrist_save = cv2.resize(live_img, (320, 240))
                img_wrist_bgr = cv2.cvtColor(img_wrist_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{args.output_dir}/wrist_{image_count:06d}.jpg", img_wrist_bgr)
                
                # Resize and save room camera image
                img_room_save = cv2.resize(live_img_room, (320, 240))
                img_room_bgr = cv2.cvtColor(img_room_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{args.output_dir}/room_{image_count:06d}.jpg", img_room_bgr)
                
                image_count += 1
            
            # Display camera feeds if monitoring is enabled
            if args.monitor_cameras:
                img_wrist_bgr = cv2.cvtColor(live_img, cv2.COLOR_RGB2BGR)
                img_room_bgr = cv2.cvtColor(live_img_room, cv2.COLOR_RGB2BGR)
                cv2.imshow("Wrist Camera", img_wrist_bgr)
                cv2.imshow("Room Camera", img_room_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("'q' pressed, stopping execution.")
                    execution_stopped = True
                    break
        
        if execution_stopped:
            break

        action_time = time.time() - action_start_time
        execution_time = time.time() - execution_start_time
        
        if i % 10 == 0:  # Print every 10 steps
            print(f"Step {i}: Action time: {action_time:.3f}s, "
                  f"Execution time: {execution_time:.3f}s")
    
    return image_count


def run_post_execution_monitoring(robot: SO101Robot):
    """Runs a final monitoring loop until the user quits."""
    if hasattr(robot, 'is_connected') and robot.is_connected and robot.enable_camera:
        print("\nPost-execution monitoring. Robot should be in its final state.")
        print("Press 'q' in a camera window to close and exit.")
        while True:
            try:
                live_img, live_img_room = robot.get_current_img()
                img_wrist_bgr = cv2.cvtColor(live_img, cv2.COLOR_RGB2BGR)
                img_room_bgr = cv2.cvtColor(live_img_room, cv2.COLOR_RGB2BGR)
                cv2.imshow("Wrist Camera", img_wrist_bgr)
                cv2.imshow("Room Camera", img_room_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting post-execution monitoring.")
                    break
            except Exception as e:
                print(f"Error during post-execution monitoring: {e}. Exiting this loop.")
                break 
    else:
        print("Skipping post-execution monitoring as robot is not connected or cameras are not enabled.")


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    print(f"Task: {args.task_description}")
    print(f"Actions to execute: {args.actions_to_execute}")
    print(f"Action horizon: {args.action_horizon}")

    # Initialize components
    client = Gr00tSO101InferenceClient(
        host=args.host,
        port=args.port,
        language_instruction=args.task_description
    )
    robot = SO101Robot(
        port_follower=args.port_follower,
        calibrate=args.calibrate,
        enable_camera=True,
    )
    setup_image_recording(args)

    try:
        with robot.activate():
            # Optional camera preview before starting
            if args.monitor_cameras:
                should_continue = run_camera_preview(robot)
                if not should_continue:
                    return

            # Run the main policy execution
            image_count = execute_policy_loop(robot, client, args)
            print("\nPolicy execution completed!")
            
            # Return to home position
            print("Returning to home position...")
            robot.go_home()
            
            # Final monitoring phase
            if args.monitor_cameras:
                run_post_execution_monitoring(robot)

    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        # In a real app, you might use logging.exception(e) for a full traceback
        # For this script, we'll proceed to the finally block for cleanup.
    finally:
        print("Cleaning up and closing windows...")
        # The 'with' statement ensures robot.disconnect() is called.
        # We just need to close the windows.
        if 'args' in locals() and args.monitor_cameras:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
