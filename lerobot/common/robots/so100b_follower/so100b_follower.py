# lerobot/common/robots/so100b_follower/so100b_follower.py
"""
SO-100B Follower + rail linéaire (Modbus-RTU).

- Bras : 6 DoF Feetech STS3215 (bus TTL)
- Rail : NEMA17 + MKS SERVO42D (bus Modbus-RTU)
"""

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any, Dict

import numpy as np

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorNormMode, MotorCalibration
from lerobot.common.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.common.motors.modbus_rtu import ModbusRTUMotorsBus
from lerobot.common.robots.utils import ensure_safe_goal_position

from .config_so100b_follower import SO100BFollowerConfig
from ..robot import Robot

logger = logging.getLogger(__name__)


class SO100BFollower(Robot):
    """Bras SO-100B + rail linéaire."""

    config_class = SO100BFollowerConfig
    name = "so100b_follower"

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #
    def __init__(self, config: SO100BFollowerConfig):
        super().__init__(config)
        self.config = config

        nm = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # ------------------------  bras Feetech ------------------------
        self.arm_bus = FeetechMotorsBus(
            port=config.port,
            motors={
                "shoulder_pan":   Motor(1, "sts3215", nm),
                "shoulder_lift":  Motor(2, "sts3215", nm),
                "elbow_flex":     Motor(3, "sts3215", nm),
                "wrist_flex":     Motor(4, "sts3215", nm),
                "wrist_roll":     Motor(5, "sts3215", nm),
                "gripper":        Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        # ------------------------  rail Modbus -------------------------
        rail_cfg        = config.rail
        self.rail_bus   = ModbusRTUMotorsBus(rail_cfg)
        self.rail_name  = next(iter(rail_cfg.motors))            # « rail »

        # ------------------------  caméras -----------------------------
        self.cameras = make_cameras_from_configs(config.cameras)

    # ------------------------------------------------------------------ #
    # États / propriétés                                                 #
    # ------------------------------------------------------------------ #
    @property
    def is_connected(self) -> bool:                               # noqa: D401
        return (
            self.arm_bus.is_connected
            and self.rail_bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:                              # noqa: D401
        return self.arm_bus.is_calibrated

    # ------------------------------------------------------------------ #
    # Espaces action / observation                                       #
    # ------------------------------------------------------------------ #
    @cached_property
    def observation_features(self) -> Dict[str, type | tuple]:
        feats = {f"{m}.pos": float for m in self.arm_bus.motors}
        feats["rail.pos"] = float
        for cam, cfg in self.config.cameras.items():
            feats[cam] = (cfg.height, cfg.width, 3)
        return feats

    @cached_property
    def action_features(self) -> Dict[str, type]:
        feats = {f"{m}.pos": float for m in self.arm_bus.motors}
        feats["rail.pos"] = float
        return feats

    # ------------------------------------------------------------------ #
    # Connexion / déconnexion                                            #
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.arm_bus.connect()
        self.rail_bus.connect()

        if not self.arm_bus.is_calibrated:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info("%s connected.", self)

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.arm_bus.disconnect(self.config.disable_torque_on_disconnect)
        self.rail_bus.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("%s disconnected.", self)

    # ------------------------------------------------------------------ #
    # Calibration du bras                                                #
    # ------------------------------------------------------------------ #
    def calibrate(self) -> None:
        logger.info("Running calibration of %s (arm only)", self)

        self.arm_bus.disable_torque()
        for m in self.arm_bus.motors:
            self.arm_bus.write("Operating_Mode", m, OperatingMode.POSITION.value)

        input("Place every joint roughly at mid-range, then press ENTER…")

        homing_offsets = self.arm_bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [m for m in self.arm_bus.motors if m != full_turn_motor]

        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their entire "
            "ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.arm_bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.arm_bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.arm_bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info("Calibration saved to %s", self.calibration_fpath)
    
    def disable_torque(self) -> None:
        """Désactive le couple des moteurs du bras et du rail."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.rail_bus.disable_torque()
        logger.info("Torque disabled for %s", self)

    # ------------------------------------------------------------------ #
    # Configuration runtime (PID, modes…)                                #
    # ------------------------------------------------------------------ #
    def configure(self) -> None:
        """Implémentation requise par Robot – configure bras & rail."""
        # -------- bras Feetech
        with self.arm_bus.torque_disabled():
            self.arm_bus.configure_motors()
            for m in self.arm_bus.motors:
                self.arm_bus.write("Operating_Mode", m, OperatingMode.POSITION.value)
                self.arm_bus.write("P_Coefficient", m, 16)
                self.arm_bus.write("I_Coefficient", m, 0)
                self.arm_bus.write("D_Coefficient", m, 32)
        # -------- rail (rien de spécial)

    # ------------------------------------------------------------------ #
    # Observation                                                        #
    # ------------------------------------------------------------------ #
    def get_observation(self) -> Dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs: Dict[str, Any] = {
            f"{m}.pos": v
            for m, v in self.arm_bus.sync_read("Present_Position").items()
        }

        # -------- rail
        try:
            rail_pos_arr = self.rail_bus.read("Present_Position")
            obs["rail.pos"] = float(rail_pos_arr[0])
        except Exception as exc:
            logger.warning("Modbus read error (%s) — rail.pos=nan", exc)
            obs["rail.pos"] = float("nan")

        print("obs rail pos", obs)

        # -------- caméras
        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()

        return obs

    # ------------------------------------------------------------------ #
    # Action                                                             #
    # ------------------------------------------------------------------ #
    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # -------- bras
        arm_goal = {
            k.removesuffix(".pos"): v
            for k, v in action.items()
            if k.endswith(".pos") and k != "rail.pos"
        }

        print("arm_goal", arm_goal)
        print("action", action)

        if self.config.max_relative_target is not None:
            present = self.arm_bus.sync_read("Present_Position")
            arm_goal = ensure_safe_goal_position(
                {m: (g, present[m]) for m, g in arm_goal.items()},
                self.config.max_relative_target,
            )

        self.arm_bus.sync_write("Goal_Position", arm_goal)

        # -------- rail
        if "rail.pos" in action and not np.isnan(action["rail.pos"]):
            try:
                self.rail_bus.write("Goal_Position", action["rail.pos"])
            except Exception as exc:
                logger.error("Modbus write error (rail): %s", exc)

        return action
