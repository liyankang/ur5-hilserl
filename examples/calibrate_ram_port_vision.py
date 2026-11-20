"""从实测样本拟合 RAM 双相机标定。此工具只读图片或相机帧。"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _numbers(values, size, field):
    if (
        not isinstance(values, list)
        or len(values) != size
        or any(type(value) not in (int, float) for value in values)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"{field} 需要 {size} 个有限数值")
    return np.asarray(values, dtype=np.float64)


def _xy_m(value, field):
    xy = _numbers(value, 2, field)
    if np.any(np.abs(xy) > 2.0):
        raise ValueError(f"{field} 超过 2 m，请检查是否误填了毫米")
    return xy


def _rect(value, field, width, height):
    x, y, w, h = _numbers(value, 4, field)
    if (
        np.any(np.floor([x, y, w, h]) != [x, y, w, h])
        or min(x, y) < 0
        or min(w, h) <= 0
        or x + w > width
        or y + h > height
    ):
        raise ValueError(f"{field} 需要图像内的整数 [x, y, width, height]")
    return [int(number) for number in (x, y, w, h)]


def _read_image(path):
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) if data else None
    if image is None:
        raise ValueError(f"无法读取原始图像 {path}")
    return image


def _camera_template(section, base_dir, template_name):
    camera = section["camera"]
    if not isinstance(camera, str) or not camera.strip():
        raise ValueError("camera 需要任务配置中的相机逻辑名")
    image_path = section.get("image")
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError(f"{camera}.image 需要填写实拍参考图的路径")
    image = _read_image(base_dir / image_path)
    height, width = image.shape[:2]
    x, y, w, h = _rect(section["template_roi"], "template_roi", width, height)
    roi_value = section.get("roi")
    roi = [0, 0, width, height] if roi_value is None else _rect(roi_value, "roi", width, height)
    rx, ry, rw, rh = roi
    if x < rx or y < ry or x + w > rx + rw or y + h > ry + rh:
        raise ValueError("template_roi 需完整位于搜索 roi 内")

    template = image[y:y + h, x:x + w]
    if min(w, h) < 3 or np.std(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)) < 2.0:
        raise ValueError("模板缺少可追踪的纹理，请框选更有辨识度的特征")
    controls = {
        "camera": camera,
        "template": template_name,
        "roi": roi,
        "frame_size_wh": [width, height],
    }
    for key, default, maximum in (("min_score", 0.8, 1.0), ("min_peak_gap", 0.08, 2.0)):
        value = _numbers([section.get(key, default)], 1, key)[0]
        if not 0 < value <= maximum:
            raise ValueError(f"{key} 必须大于 0 且不超过 {maximum}")
        controls[key] = float(value)
    center = np.array([x + (w - 1) / 2, y + (h - 1) / 2])
    return controls, template, center


def _measured_pairs(samples, xy_key, roi, minimum):
    if not isinstance(samples, list) or len(samples) < minimum:
        raise ValueError(f"samples 至少需要 {minimum} 组有效测量")
    pixels = np.array([_numbers(sample["pixel_xy"], 2, "pixel_xy") for sample in samples])
    ground = np.array([_xy_m(sample[xy_key], xy_key) for sample in samples])
    x, y, width, height = roi
    if np.any(pixels < [x, y]) or np.any(pixels > [x + width - 1, y + height - 1]):
        raise ValueError("样本特征中心必须位于搜索 roi 内，请使用原始全图像素")
    return pixels, ground


def _check_error(error, threshold, units):
    limit = _numbers([threshold], 1, "max_fit_error")[0]
    maximum = float(np.max(np.linalg.norm(error, axis=1)))
    if not np.isfinite(maximum) or not 0 < limit or maximum > limit:
        raise ValueError(f"拟合最大残差 {maximum:.6g} {units} 超过允许值 {limit:g} {units}")
    return maximum


def _fit_fixed(section, roi):
    pixels, base = _measured_pairs(section["samples"], "base_xy_m", roi, 4)
    homogeneous = np.column_stack([pixels, np.ones(len(pixels))])
    zeros = np.zeros_like(homogeneous)
    design = np.vstack([
        np.hstack([homogeneous, zeros, -base[:, :1] * homogeneous]),
        np.hstack([zeros, homogeneous, -base[:, 1:] * homogeneous]),
    ])
    if np.linalg.matrix_rank(design) < 8:
        raise ValueError("固定相机测量点无法唯一拟合单应矩阵，请覆盖 XY 平面")
    mapping, _ = cv2.findHomography(pixels, base, method=0)
    if mapping is None or not np.all(np.isfinite(mapping)) or np.linalg.matrix_rank(mapping) < 3:
        raise ValueError("固定相机单应矩阵无效")
    x, y, width, height = roi
    corners = np.array([
        [x, y, 1], [x + width - 1, y, 1],
        [x, y + height - 1, 1], [x + width - 1, y + height - 1, 1],
    ])
    scales = corners @ mapping[2]
    if not (np.all(scales > 1e-9) or np.all(scales < -1e-9)):
        raise ValueError("固定相机映射在搜索区域内存在奇异点")
    projected = cv2.perspectiveTransform(pixels[None], mapping)[0]
    error = _check_error(projected - base, section.get("max_fit_error_m", 0.005), "m")
    return mapping, error


def _fit_wrist(section, roi, reference_pixel):
    pixels, tcp_xy = _measured_pairs(section["samples"], "tcp_xy_m", roi, 2)
    offsets = tcp_xy - _xy_m(section["reference_tcp_xy_m"], "reference_tcp_xy_m")
    pixel_motion = pixels - reference_pixel
    coefficients, _, rank, _ = np.linalg.lstsq(offsets, pixel_motion, rcond=None)
    jacobian = coefficients.T
    if (
        rank < 2
        or not np.all(np.isfinite(jacobian))
        or np.linalg.cond(jacobian) > 1000.0
    ):
        raise ValueError("腕部雅可比矩阵退化，请分别采集基座 X 和 Y 移动的测量")
    error = _check_error(offsets @ coefficients - pixel_motion, section.get("max_fit_error_px", 2.0), "px")
    return jacobian, error


def _encode_png(image):
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("图像编码为 PNG 失败")
    return png.tobytes()


def _write_new_files(files):
    created = []
    try:
        for path, data in files:
            with path.open("xb") as output:
                created.append(path)
                output.write(data)
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise


def fit(measurements_path, output_path):
    measurements = json.loads(measurements_path.read_text(encoding="utf-8-sig"))
    approach_z = _numbers([measurements["approach_z_m"]], 1, "approach_z_m")[0]
    if abs(approach_z) > 2.0:
        raise ValueError("approach_z_m 超过 2 m，请检查是否误填了毫米")
    fixed, fixed_png, _ = _camera_template(
        measurements["fixed"], measurements_path.parent, f"{output_path.stem}_fixed.png"
    )
    wrist, wrist_png, reference = _camera_template(
        measurements["wrist"], measurements_path.parent, f"{output_path.stem}_wrist.png"
    )
    if fixed["camera"] == wrist["camera"]:
        raise ValueError("固定相机和腕部相机需要不同的逻辑名")
    homography, fixed_error = _fit_fixed(measurements["fixed"], fixed["roi"])
    jacobian, wrist_error = _fit_wrist(measurements["wrist"], wrist["roi"], reference)
    fixed["pixel_to_base_homography"] = homography.tolist()
    wrist["pixel_per_meter_jacobian"] = jacobian.tolist()
    wrist["reference_pixel_xy"] = reference.tolist()
    config = {"approach_z_m": float(approach_z), "fixed": fixed, "wrist": wrist}

    # 部署 JSON 最后写入，路径相对它解析。已有文件通过排他创建保护。
    _write_new_files([
        (output_path.parent / fixed["template"], _encode_png(fixed_png)),
        (output_path.parent / wrist["template"], _encode_png(wrist_png)),
        (output_path, (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")),
    ])
    return {"config": str(output_path), "fixed_fit_max_error_m": fixed_error, "wrist_fit_max_error_px": wrist_error}


def capture(device, output_path, width, height, fps):
    from ur_env.camera.v4l_capture import V4LCapture

    if output_path.suffix.lower() != ".png" or min(width, height, fps) <= 0:
        raise ValueError("请指定 PNG 输出和正数分辨率、帧率")
    with output_path.open("xb") as output:
        try:
            camera = V4LCapture(name="calibration", video_device=device, dim=(width, height), fps=fps)
            try:
                for _ in range(5):
                    ok, image = camera.read()
                    if not ok or image is None:
                        raise RuntimeError(f"摄像头 {device} 未返回图像")
            finally:
                camera.close()
            if image.shape[:2] != (height, width):
                raise ValueError(f"摄像头的实际分辨率是 {image.shape[1]}x{image.shape[0]}")
            output.write(_encode_png(image))
        except BaseException:
            output.close()
            output_path.unlink()
            raise
    return {"image": str(output_path), "frame_size_wh": [width, height]}


def inspect_frames(config_path, fixed_image=None, wrist_image=None):
    from examples.experiments.ram_insertion.port_vision import PortVision

    vision = PortVision.load(config_path)
    report = {"approach_z_m": vision.approach_z_m}
    for stage, image, estimate, xy_name in (
        ("fixed", fixed_image, vision.estimate_port_xy, "base_xy_m"),
        ("wrist", wrist_image, vision.wrist_correction, "tcp_delta_xy_m"),
    ):
        if image is None:
            continue
        result = estimate(_read_image(image))
        report[stage] = {
            "pixel_xy": result.pixel_xy.tolist(),
            xy_name: result.xy_m.tolist(),
            "score": result.score,
            "second_score": result.second_score,
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit", help="读取现场测量 JSON，拟合部署配置")
    fit_parser.add_argument("--measurements", required=True, type=Path)
    fit_parser.add_argument("--output", required=True, type=Path)
    capture_parser = subparsers.add_parser("capture", help="只通过相机保存原始 PNG")
    capture_parser.add_argument("--device", required=True, help="与任务配置相同的 /dev/videoN")
    capture_parser.add_argument("--output", required=True, type=Path)
    capture_parser.add_argument("--width", default=1280, type=int)
    capture_parser.add_argument("--height", default=720, type=int)
    capture_parser.add_argument("--fps", default=10, type=int)
    inspect_parser = subparsers.add_parser("inspect", help="从保存的图像检查定位或腕部修正")
    inspect_parser.add_argument("--config", required=True, type=Path)
    inspect_parser.add_argument("--fixed-image", type=Path)
    inspect_parser.add_argument("--wrist-image", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "fit":
            report = fit(args.measurements, args.output)
        elif args.command == "capture":
            report = capture(args.device, args.output, args.width, args.height, args.fps)
        else:
            if args.fixed_image is None and args.wrist_image is None:
                inspect_parser.error("至少指定一张 --fixed-image 或 --wrist-image")
            report = inspect_frames(args.config, args.fixed_image, args.wrist_image)
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, cv2.error) as error:
        parser.exit(2, f"标定工具失败：{error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
