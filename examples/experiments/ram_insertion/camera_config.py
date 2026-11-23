"""现场相机的设备路径和 RAM 策略检查点使用的图像键。"""

RAM_CAMERAS = {
    "global_1": {"video_device": "/dev/video2", "dim": (1280, 720)},
    "wrist": {"video_device": "/dev/video4", "dim": (1280, 720)},
    "global_2": {"video_device": "/dev/video0", "dim": (1280, 720)},
}
