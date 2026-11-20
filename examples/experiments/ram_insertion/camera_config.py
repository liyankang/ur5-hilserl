"""RAM 检查点使用的图像键，以及现场双相机标定的物理角色。"""

RAM_CAMERAS = {
    "global_1": {"video_device": "/dev/video2", "dim": (1280, 720)},
    "wrist": {"video_device": "/dev/video4", "dim": (1280, 720)},
    "global_2": {"video_device": "/dev/video0", "dim": (1280, 720)},
}

PORT_CAMERA_KEYS = {"fixed": "wrist", "wrist": "global_2"}


def validate_port_camera_roles(fixed_camera, wrist_camera):
    if fixed_camera == wrist_camera:
        raise ValueError("固定相机和腕部相机需要不同的逻辑名")
    if (fixed_camera, wrist_camera) == (PORT_CAMERA_KEYS["wrist"], PORT_CAMERA_KEYS["fixed"]):
        raise ValueError(
            "RAM 标定的相机角色填反了。固定相机应为 wrist（/dev/video4），"
            "腕部相机应为 global_2（/dev/video0）。请用对应的参考图和样本重新标定。"
        )
