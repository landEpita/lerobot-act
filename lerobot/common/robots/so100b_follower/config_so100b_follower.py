# lerobot/common/robots/config_so100b_follower.py
from dataclasses import dataclass, field
from lerobot.common.cameras import CameraConfig

# NEW correct import path
from lerobot.common.motors.modbus_rtu import ModbusRTUMotorsBusConfig
from ..config import RobotConfig


@RobotConfig.register_subclass("so100b_follower")
@dataclass
class SO100BFollowerConfig(RobotConfig):
    """Configuration for the SO‑100B follower arm + linear rail."""

    # Feetech 6‑DoF arm
    port: str

    # Linear rail (Modbus‑RTU)
    rail: ModbusRTUMotorsBusConfig

    # Cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Safety
    max_relative_target: int | None = None
    disable_torque_on_disconnect: bool = True

    # Units
    use_degrees: bool = False