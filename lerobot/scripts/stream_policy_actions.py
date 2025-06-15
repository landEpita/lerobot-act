"""
Run a *pre‑trained* policy on the robot (no dataset, no sim) **with optional one‑hot task vector**.

Usage examples
--------------
* **Single task** (no one‑hot):

    python run_robot_actions.py

* **Multi‑task** with one‑hot `[0, 1]`:

    python run_robot_actions.py --task 0 1

Everything else (robot config, checkpoint path, FPS…) is editable below.
"""

import time
from collections import deque
from dataclasses import dataclass

import torch, os

from lerobot.common.policies.factory import get_policy_class
from lerobot.common.robot_devices.control_utils import predict_action
from lerobot.common.robot_devices.motors.configs import ModbusRTUMotorsBusConfig
from lerobot.common.robot_devices.motors.modbus_rtu_motor import ModbusRTUMotorsBus

########################################################################################
# USER‑EDITABLE SECTION                                                                #
########################################################################################
# 1. Provide your own `robot_cfg` (example below).
# 2. Point `PRETRAINED_PATH` to your checkpoint.
# 3. Adjust runtime parameters (FPS, EPISODE_TIME_S, etc.).
########################################################################################
# --- Robot configuration -------------------------------------------------------------
from lerobot.common.robot_devices.robots.configs import (
    FeetechMotorsBusConfig,
    MonRobot7AxesConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import get_safe_torch_device

robot_cfg = MonRobot7AxesConfig(
    # leader_arms={
    #     "left": FeetechMotorsBusConfig(
    #         port="/dev/tty.usbmodem58FD0166391",
    #         motors={
    #             "shoulder_pan": [1, "sts3215"],
    #             "shoulder_lift": [2, "sts3215"],
    #             "elbow_flex": [3, "sts3215"],
    #             "wrist_flex": [4, "sts3215"],
    #             "wrist_roll": [5, "sts3215"],
    #             "gripper": [6, "sts3215"],
    #         },
    #         mock=False,
    #     )
    # },
    follower_arms={
        "left": FeetechMotorsBusConfig(
            port="/dev/tty.usbmodem58FD0162241",
            motors={
                "shoulder_pan": [1, "sts3215"],
                "shoulder_lift": [2, "sts3215"],
                "elbow_flex": [3, "sts3215"],
                "wrist_flex": [4, "sts3215"],
                "wrist_roll": [5, "sts3215"],
                "gripper": [6, "sts3215"],
            },
            mock=False,
        ),
        "rail_lineaire": ModbusRTUMotorsBusConfig(
            port="/dev/tty.usbserial-BG00Q7CQ",
            motors={"axe_translation": (1, "NEMA17_MKS42D")},
            baudrate=115200,
        ),
    },
    cameras={
        "webcam": OpenCVCameraConfig(
            camera_index=0,
            fps=30,
            width=640,
            height=480,
            color_mode="rgb",
            channels=3,
            rotation=None,
            mock=False,
        ),
        "camD": OpenCVCameraConfig(
            camera_index=1,
            fps=30,
            width=640,
            height=480,
            color_mode="rgb",
            channels=3,
            rotation=None,
            mock=False,
        ),
    },
    max_relative_target=None,
    gripper_open_degree=None,
    mock=False,
    calibration_dir=".cache/calibration/so100b",
)

# --- Policy --------------------------------------------------------------------------
PRETRAINED_PATH = "/Users/thomas/Documents/lbc/robot/lerobot-act/model/act_final-task-4_140k"  # <-- change me
# PRETRAINED_PATH = "/Users/thomas/Documents/lbc/robot/lerobot-act/model/task_robot_1_40k"  # <-- change me
POLICY_TYPE = "act"  # "tdmpc", "diffusion", …
DEVICE = "mps"  # | "cpu" | "mps"

########################################################################################
# RUNTIME PARAMETERS                                                                   #
########################################################################################
FPS = 30
EPISODE_TIME_S = 50  # seconds — set to None for infinite runtime

########################################################################################
# CONTROL LOOP                                                                         #
########################################################################################


@dataclass
class ControlParams:
    fps: int = FPS
    episode_time_s: float | None = EPISODE_TIME_S
    onehot: torch.Tensor | None = None


@safe_disconnect
def run_actions(robot: Robot, params: ControlParams, policy) -> None:
    """Stream actions from a pre‑trained policy, optionally adding a one‑hot task vector."""

    # --- Connect robot ---
    if not robot.is_connected:
        robot.connect()

    # Enable torque on follower arms if needed
    for arm in robot.follower_arms.values():
        if isinstance(arm, ModbusRTUMotorsBus):
            arm.write("Torque_Enable", 1)

    # --- Load policy once ---
    device = get_safe_torch_device(DEVICE)

    fifo: deque[torch.Tensor] = deque(maxlen=20)
    per_axis_thresh = torch.tensor([0.5, 0.5, 0.7, 0.7, 0.7, 0.1, 1500])

    start_episode_t = time.perf_counter()
    timestamp = 0.0
    start_time = time.time()

    print(">>> Running policy…  (Ctrl‑C to stop)")
    while params.episode_time_s is None or timestamp < params.episode_time_s:
        loop_start_t = time.perf_counter()

        # 1) Observation ↦ policy ↦ action
        obs = robot.capture_observation()
        if params.onehot is not None:
            obs["onehot_task"] = params.onehot
        act = predict_action(obs, policy, device, use_amp=False)
        sent_act = robot.send_action(act)
        fifo.append(sent_act.clone())

        # 2) Auto‑stop if no movement
        if len(fifo) == fifo.maxlen:
            std_per_motor = torch.std(torch.stack(list(fifo)), dim=0)
            if torch.all(std_per_motor < per_axis_thresh):
                if (time.time() - start_time) > 5:
                    print(">>> Auto‑stop: robot idle (std < threshold)")
                    break

        # 3) Keep constant FPS
        if params.fps:
            busy_wait(max(0, 1 / params.fps - (time.perf_counter() - loop_start_t)))

        timestamp = time.perf_counter() - start_episode_t

    print(">>> Done!")


########################################################################################
# CLI / ENTRY POINT                                                                    #
########################################################################################

PATH = os.path.dirname(os.path.abspath(__file__))

def get_policies():
    policy_cls = get_policy_class(POLICY_TYPE)
    policy_0 = policy_cls.from_pretrained(os.path.join("model/act_final-task-0_140k")).to(DEVICE)
    policy_1 = policy_cls.from_pretrained(os.path.join("model/act_final-task-1_140k")).to(DEVICE)
    policy_2 = policy_cls.from_pretrained(os.path.join("model/act_final-task-2_140k")).to(DEVICE)
    policy_3 = policy_cls.from_pretrained(os.path.join("model/act_final-task-3_140k")).to(DEVICE)
    policy_4 = policy_cls.from_pretrained(os.path.join("model/act_final-task-4_140k")).to(DEVICE)
    return [policy_0, policy_1, policy_2, policy_3, policy_4]

def get_robot():
    return make_robot_from_config(robot_cfg)

def run_action(policy, robot):
    params = ControlParams()
    run_actions(robot, params, policy)


if __name__ == "__main__":
    policy_cls = get_policy_class(POLICY_TYPE)
    policy_0 = policy_cls.from_pretrained("/Users/thomas/Documents/lbc/robot/lerobot-act/model/act_final-task-0_140k").to(DEVICE)
    policy_1 = policy_cls.from_pretrained("/Users/thomas/Documents/lbc/robot/lerobot-act/model/act_final-task-1_140k").to(DEVICE)
    policy_2 = policy_cls.from_pretrained("/Users/thomas/Documents/lbc/robot/lerobot-act/model/act_final-task-2_140k").to(DEVICE)
    policy_3 = policy_cls.from_pretrained("/Users/thomas/Documents/lbc/robot/lerobot-act/model/act_final-task-3_140k").to(DEVICE)
    policy_4 = policy_cls.from_pretrained("/Users/thomas/Documents/lbc/robot/lerobot-act/model/act_final-task-4_140k").to(DEVICE)
    policy_act = policy_cls.from_pretrained("/Users/thomas/Documents/lbc/robot/lerobot-act/model/leo-3").to(DEVICE)
    

    onehot_tensor = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float)

    params = ControlParams(onehot=onehot_tensor)
    # params = ControlParams()
    robot = make_robot_from_config(robot_cfg)
    polices = [policy_act]
    for i, policy in enumerate(polices):
        print(f"Running policy {i}...")
        run_actions(robot, params, policy)
        time.sleep(1)  # Small delay between policies
