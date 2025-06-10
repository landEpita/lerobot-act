from lerobot.common.motors.modbus_rtu import ModbusRTUMotorsBusConfig
from lerobot.common.teleoperators.so100_leader import SO100LeaderConfig, SO100Leader
from lerobot.common.robots.so100b_follower import SO100BFollowerConfig, SO100BFollower

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.common.cameras.configs import ColorMode, Cv2Rotation


robot_config = SO100BFollowerConfig(
    id="desk_arm",
    port="/dev/tty.usbmodem58FD0172321",
    rail=ModbusRTUMotorsBusConfig(
        port="/dev/tty.usbserial-BG00Q7CQ",
        motors={"rail": (1, "NEMA17_MKS42D")},
    ),
)

teleop_config = SO100LeaderConfig(
    port="/dev/tty.usbmodem58FD0166391",
    id="my_blue_leader_arm",
)

configWebcam = OpenCVCameraConfig(
    index_or_path=0,
    fps=30,
    width=1920,
    height=1080,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
)

configLeft = OpenCVCameraConfig(
    index_or_path=1,
    fps=30,
    width=1920,
    height=1080,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
)

cameraWebcam = OpenCVCamera(configWebcam)
cameraLeft = OpenCVCamera(configLeft)
cameraWebcam.connect()
cameraLeft.connect()

robot = SO100BFollower(robot_config)
teleop_device = SO100Leader(teleop_config)

robot.connect()   # ouvre les ports seulement

teleop_device.connect()   # ouvre les ports seulement

# robot.connect()

while True:
    action = teleop_device.get_action()
    robot.send_action(action)