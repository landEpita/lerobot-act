import logging
import os
import time
from dataclasses import asdict, dataclass
from pprint import pformat
import numpy as np
import random
import shutil
import rerun as rr
import collections

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    RecordControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    sanity_check_dataset_name,
    warmup_record,
    init_keyboard_listener,
    predict_action,
)
from lerobot.common.robot_devices.motors.configs import ModbusRTUMotorsBusConfig
from lerobot.common.robot_devices.motors.modbus_rtu_motor import ModbusRTUMotorsBus
from lerobot.common.robot_devices.robots.configs import (
    FeetechMotorsBusConfig,
    MonRobot7AxesConfig,
    OpenCVCameraConfig,
    So100RobotConfig,
)
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import safe_disconnect
from lerobot.common.policies.act.configuration_act import (
    ACTConfig,
    NormalizationMode,
)
from lerobot.configs.types import PolicyFeature, FeatureType

import shutil
import torch


########################################################################################
# Control modes
########################################################################################

@safe_disconnect
def teleoperate(robot: Robot, cfg: TeleoperateControlConfig):
    control_loop(
        robot,
        control_time_s=cfg.teleop_time_s,
        fps=cfg.fps,
        teleoperate=True,
        display_data=cfg.display_data,
    )


@dataclass
class Config:
    robot: So100RobotConfig
    control: RecordControlConfig

@safe_disconnect
def record(
    robot: Robot,
    cfg: RecordControlConfig,
    index:int,
) -> LeRobotDataset:
    cfg.repo_id = cfg.repo_id + "_" + str(index)
    # Create empty dataset or load existing saved episodes
    sanity_check_dataset_name(cfg.repo_id, cfg.policy)
    dataset = LeRobotDataset.create(
        cfg.repo_id,
        cfg.fps,
        root=cfg.root,
        robot=robot,
        use_videos=cfg.video,
        image_writer_processes=cfg.num_image_writer_processes,
        image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
    )

    # Load pretrained policy
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    if not robot.is_connected:
        robot.connect()

    for name in robot.follower_arms:
        print("name: ", name)
        if isinstance(robot.follower_arms[name], ModbusRTUMotorsBus):
            robot.follower_arms[name].write("Torque_Enable", 1)

    control_time_s = cfg.episode_time_s

    if control_time_s is None:
        control_time_s = float("inf")

    if dataset is not None and cfg.fps is not None and dataset.fps != cfg.fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {cfg.fps}).")
    
    timestamp = 0
    start_episode_t = time.perf_counter()
    memory  = collections.deque(maxlen=10)
    last_value = None
    # print("@@@@@@ Start recording trajectory @@@@@@")
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        
        observation = robot.capture_observation()
        # print("observation: ", observation)
        if policy is not None:
            pred_action = predict_action(
                observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
            )
            # print("Action sent: ", pred_action[-1])

            memory.append(pred_action[-1])
            if len(memory) > 5:
                pred_action = pred_action.clone()  # Detach from inference mode
                moyenne = sum(memory) / len(memory)
                if last_value is not None and abs(moyenne - last_value) < 1000 :
                    print("Action last: ", last_value)
                    pred_action[-1] = last_value
                else:
                    pred_action[-1] = sum(memory) / len(memory)
                    print("Action smoothed: ", pred_action[-1])
                    last_value = pred_action[-1]
            else:
                print("Action sent: ", pred_action[-1])

                
            

            # Action can eventually be clipped using `max_relative_target`,
            # so action actually sent is saved in the dataset.
            action = robot.send_action(pred_action)
            action = {"action": action}


        if cfg.fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / cfg.fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        timestamp = time.perf_counter() - start_episode_t
    print("Finished recording trajectory")


@dataclass
class Config_dummy():
    robot: MonRobot7AxesConfig
    control: RecordControlConfig

def control_robot(
    index: int,
):
    # TODO : fix the call to config here 
    cfg = Config_dummy(
        robot=MonRobot7AxesConfig(
            leader_arms={
                'left': FeetechMotorsBusConfig(
                    port="/dev/tty.usbmodem58FD0166391",
                    motors={
                        'shoulder_pan': [1, 'sts3215'],
                        'shoulder_lift': [2, 'sts3215'],
                        'elbow_flex': [3, 'sts3215'],
                        'wrist_flex': [4, 'sts3215'],
                        'wrist_roll': [5, 'sts3215'],
                        'gripper': [6, 'sts3215']
                    },
                    mock=False
                )
            },
            follower_arms={
                'left': FeetechMotorsBusConfig(
                    port="/dev/tty.usbmodem58FD0162261",
                    motors={
                        'shoulder_pan': [1, 'sts3215'],
                        'shoulder_lift': [2, 'sts3215'],
                        'elbow_flex': [3, 'sts3215'],
                        'wrist_flex': [4, 'sts3215'],
                        'wrist_roll': [5, 'sts3215'],
                        'gripper': [6, 'sts3215']
                    },
                    mock=False
                ),
                "rail_lineaire": ModbusRTUMotorsBusConfig( # Votre axe NEMA17
                    port="/dev/tty.usbserial-BG00Q7CQ", # Adaptez
                    motors={"axe_translation": (1, "NEMA17_MKS42D")}, # Nom du moteur et son ID Modbus
                    baudrate=115200,
                ),
            },
            cameras={
                'webcam': OpenCVCameraConfig(
                    camera_index=0,
                    fps=30,
                    width=640,
                    height=480,
                    color_mode='rgb',
                    channels=3,
                    rotation=None,
                    mock=False
                ),
                # 'camD': OpenCVCameraConfig(
                #     camera_index=1,
                #     fps=30,
                #     width=640,
                #     height=480,
                #     color_mode='rgb',
                #     channels=3,
                #     rotation=None,
                #     mock=False
                # )
            },
            max_relative_target=None,
            gripper_open_degree=None,
            mock=False,
            calibration_dir='.cache/calibration/so100b'
        ), 
        control=RecordControlConfig(
            repo_id='tgossin/eval_so100_dataset_mks',
            single_task='Grasp a lego block and put it in the bin.',
            # multi_task=True,
            root=None,
            policy=ACTConfig(
                n_obs_steps=1,
                normalization_mapping={
                    'VISUAL': NormalizationMode.MEAN_STD,
                    'STATE': NormalizationMode.MEAN_STD,
                    'ACTION': NormalizationMode.MEAN_STD
                },
                input_features={
                    'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(7,)),
                    'observation.images.mounted': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
                },
                output_features={
                    'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))
                },
                device='cuda',
                use_amp=False,
                chunk_size=100,
                n_action_steps=100,
                # use_onehot=True,
                # onehot_action_dim=3,
                vision_backbone='resnet18',
                pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1',
                replace_final_stride_with_dilation=0,
                pre_norm=False,
                dim_model=512,
                n_heads=8,
                dim_feedforward=3200,
                feedforward_activation='relu',
                n_encoder_layers=4,
                n_decoder_layers=1,
                use_vae=True,
                latent_dim=32,
                n_vae_encoder_layers=4,
                temporal_ensemble_coeff=None,
                dropout=0.1,
                kl_weight=10.0,
                optimizer_lr=1e-05,
                optimizer_weight_decay=0.0001,
                optimizer_lr_backbone=1e-05,
                # pretrained_path='/Users/thomas/Documents/lbc/robot/lerobot-act/model/mkd'
            ),
            fps=30,
            warmup_time_s=5,
            episode_time_s=50,
            reset_time_s=8,
            num_episodes=1,
            video=True,
            push_to_hub=False,
            private=False,
            tags=[''],
            num_image_writer_processes=0,
            num_image_writer_threads_per_camera=4,
            display_data=False,
            play_sounds=True,
            resume=False
        )
    )
    cfg.control.policy.pretrained_path = '/Users/thomas/Documents/lbc/robot/lerobot/model/mks2'

    robot = make_robot_from_config(cfg.robot)
    record(robot, cfg.control, index=index)



if __name__ == "__main__":
    index = 0
    control_robot(index=index)