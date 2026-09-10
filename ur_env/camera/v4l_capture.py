import os
import time

import cv2


class V4LCapture:
    """Simple V4L2/OpenCV capture wrapper for standard USB cameras."""

    def __init__(self, name, video_device, dim=(640, 480), fps=10, fourcc="YUYV"):
        self.name = name
        self.video_device = video_device
        self.dim = dim
        self.fps = fps
        self.fourcc = fourcc

        source = self._resolve_source(video_device)
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open V4L2 camera {video_device!r} for {name}")

        if dim is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(dim[0]))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(dim[1]))
        if fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, float(fps))
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        time.sleep(0.3)

    @staticmethod
    def _resolve_source(video_device):
        if isinstance(video_device, int):
            return video_device
        if isinstance(video_device, str):
            base = os.path.basename(video_device)
            if base.isdigit():
                return int(base)
        return video_device

    def read(self):
        return self.cap.read()

    def close(self):
        if self.cap is not None:
            self.cap.release()
