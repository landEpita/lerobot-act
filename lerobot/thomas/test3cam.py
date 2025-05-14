from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

config = OpenCVCameraConfig(camera_index=1)
camera = OpenCVCamera(config)
camera.connect()
color_image = camera.read()

print(color_image.shape)
print(color_image.dtype)