"""
相机图像裁剪工具（静态帧版）
用法: python tests/camera_crop_viewer.py
流程: 选相机 → 拍一帧 → 鼠标拖拽画框 → 松开后右边显示裁剪结果
操作:
  鼠标拖拽 = 画裁剪框（松开后显示裁剪结果）
  r        = 重新拍一帧
  q / ESC  = 退出并打印裁剪参数
"""
import cv2, time, sys
import numpy as np

CAMERAS = {
    "wrist_1": {"device": "/dev/video0", "crop": (0, 540, 100, 500)},
    "wrist_2": {"device": "/dev/video2", "crop": (120, 500, 500, 800)},
    "wrist_3": {"device": "/dev/video4", "crop": (200, 500, 450, 700)},
}
# crop: (y1, y2, x1, x2) → img[y1:y2, x1:x2]


def select_camera():
    print("可用相机:")
    cams = list(CAMERAS.keys())
    for i, name in enumerate(cams):
        print(f"  [{i}] {name}  ({CAMERAS[name]['device']})")
    choice = input("选择相机编号 (默认 0): ").strip()
    return cams[int(choice)] if choice else cams[0]


def main():
    cam_name = select_camera()
    cfg = CAMERAS[cam_name]
    device = cfg["device"]

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"打开 {device} 失败！")
        sys.exit(1)
    time.sleep(0.3)

    def grab_frame():
        ret, frame = cap.read()
        if not ret:
            print("读取失败！")
            sys.exit(1)
        return frame

    frame = grab_frame()
    h, w = frame.shape[:2]
    print(f"原始分辨率: {w}x{h}")

    # 状态
    y1, y2, x1, x2 = cfg["crop"]
    drawing = False
    drag_start = None
    show_crop = True  # 初始显示默认裁剪

    win = f"{cam_name} - Crop Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1000, 600)

    def mouse_cb(event, mx, my, flags, param):
        nonlocal drawing, drag_start, x1, x2, y1, y2, show_crop
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            show_crop = False
            drag_start = (mx, my)
            x1, x2, y1, y2 = mx, mx, my, my
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x1, x2 = sorted([drag_start[0], mx])
            y1, y2 = sorted([drag_start[1], my])
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, x2 = sorted([drag_start[0], mx])
            y1, y2 = sorted([drag_start[1], my])
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            show_crop = True  # 松开鼠标，显示裁剪结果

    cv2.setMouseCallback(win, mouse_cb)

    print("\n操作: 鼠标拖拽画框 → 松开显示裁剪结果 | r=重新拍帧 | q/ESC=退出")

    while True:
        vis = frame.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"img[{y1}:{y2}, {x1}:{x2}]  ({x2-x1}x{y2-y1})"
        cv2.putText(vis, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if show_crop and (x2 - x1) > 5 and (y2 - y1) > 5:
            cropped = frame[y1:y2, x1:x2]
            ch, cw = cropped.shape[:2]
            disp_h = min(500, h)
            scale = disp_h / ch
            disp_w = int(cw * scale)
            cropped_disp = cv2.resize(cropped, (disp_w, disp_h))

            pad = 15
            canvas_h = max(h, disp_h + pad * 2)
            canvas_w = w + disp_w + pad * 3
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            canvas[:h, :w] = vis
            canvas[pad:pad+disp_h, w+pad*2:w+pad*2+disp_w] = cropped_disp
            cv2.putText(canvas, "Cropped Result", (w + pad * 2, pad - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            canvas = vis

        cv2.imshow(win, canvas)
        key = cv2.waitKey(50)
        if key != -1:
            key = key & 0xFF  # 只取低8位，兼容Linux
            # print(f"key={key}")  # 调试用
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('r'):
                frame = grab_frame()
                y1, y2, x1, x2 = cfg["crop"]
                show_crop = True
                print("已重新拍帧")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[{cam_name}] 最终裁剪参数（复制到 config.py IMAGE_CROP）:")
    print(f'  "{cam_name}": lambda img: img[{y1}:{y2}, {x1}:{x2}],')
    print(f"  裁剪后尺寸: {x2-x1}x{y2-y1}")


if __name__ == "__main__":
    main()
