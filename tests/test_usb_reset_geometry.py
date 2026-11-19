"""用真实 SciPy 旋转验证 USB 随机复位和退出路径，不连接相机或机械臂。

生产类通过 AST 载入，只有 RTDE/HTTP 和等待被替换；Rotation/Slerp 使用真实
实现。缺少 SciPy 时明确跳过，不能把单位旋转替身的通过当成几何验证。
"""
import ast
import logging
import math
from pathlib import Path
import types
import unittest

import numpy as np

try:
    from scipy.spatial.transform import Rotation, Slerp
except ImportError:
    Rotation = Slerp = None

from examples.experiments.usb_insertion.runtime_config import load_config, require_hardware_ready


ROOT = Path(__file__).resolve().parents[1]


def load_definition(relative, name, namespace):
    """载入实际源代码定义，避免模块顶层初始化 Gym/JAX 或硬件依赖。"""
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"))
    node = next(item for item in tree.body if getattr(item, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), namespace)
    return namespace[name]


class Clock:
    """只推进虚拟时间，保留超时逻辑而不让测试实际等待几秒。"""
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid wait duration")
        self.now += duration


class FakeRAM:
    """提供父类的数据接口，实际 USB 随机化与插值方法不做替换。"""
    def __init__(self, *, config, fake_env=False, hz=50, save_video=False):
        if not fake_env:
            raise AssertionError("Geometry tests must never create a real robot environment")
        self.config = config
        self.hz = hz
        self.save_video = save_video
        self.randomreset = config.RANDOM_RESET
        self._TARGET_POSE = config.TARGET_POSE.copy()
        self.xyz_bounding_box = types.SimpleNamespace(
            low=config.ABS_POSE_LIMIT_LOW[:3].copy(),
            high=config.ABS_POSE_LIMIT_HIGH[:3].copy(),
        )
        self.resetpos = self._pose6_to_quat(config.RESET_POSE)
        self.currpos = self.resetpos.copy()
        self.actual_pose = self.currpos.copy()
        self.commands = []
        self.hold_count = 0
        self.feedback_count = 0

    def _pose6_to_quat(self, pose):
        return np.r_[np.asarray(pose)[:3], Rotation.from_rotvec(np.asarray(pose)[3:]).as_quat()]

    def _update_currpos(self):
        self.feedback_count += 1
        self.currpos = self.actual_pose.copy()

    def _send_pos_command(self, pose, source="policy"):
        self.commands.append((np.asarray(pose).copy(), source))
        self.actual_pose = np.asarray(pose).copy()

    def _get_obs(self):
        return {"state": {"tcp_pose": self.currpos.copy()}}

    def close(self):
        pass


def make_env(seed=42):
    if Rotation is None:
        raise unittest.SkipTest("Real SciPy Rotation/Slerp is required")
    task = load_config()
    # 真实 EnvConfig.__init__ 从真实 JSON 构建全部复位与限幅参数。
    target_builder = load_definition(
        "examples/experiments/usb_insertion/config.py", "build_target_poses", {"np": np},
    )
    config_class = load_definition(
        "examples/experiments/usb_insertion/config.py", "EnvConfig",
        {"np": np, "DefaultEnvConfig": object, "load_config": load_config,
         "controller_safety_payload": lambda settings: {},
         "build_target_poses": target_builder},
    )
    config = config_class(task)
    config.RESET_SEED = seed
    clock = Clock()

    def forbidden_post(*args, **kwargs):
        raise AssertionError("Unexpected HTTP request in geometry test")

    env_class = load_definition(
        "examples/experiments/usb_insertion/wrapper.py", "USBInsertionEnv",
        {"np": np, "math": math, "time": clock,
         "Rotation": Rotation, "Slerp": Slerp, "RAMEnv": FakeRAM,
         "requests": types.SimpleNamespace(post=forbidden_post),
         "LOGGER": logging.getLogger("usb_reset_geometry_test")},
    )
    env = env_class(config=config, fake_env=True, hz=config.CONTROL_HZ)

    def hold():
        env.hold_count += 1
    # 成功/超时保持只记录调用，不向实际服务发请求。
    env.hold_position = hold
    return env


@unittest.skipUnless(Rotation is not None, "Real SciPy Rotation/Slerp is required")
class USBResetGeometryTests(unittest.TestCase):
    def test_default_random_resets_cover_xyz_and_keep_reference_orientation(self):
        env = make_env()
        target_before = env._TARGET_POSE.copy()
        poses = np.asarray([env.sample_reset_pose() for _ in range(1000)])
        nominal = env.config.RESET_POSE
        offset = poses[:, :3] - nominal[:3]
        jitter = env.config.RESET_TRANSLATION_RANGE_M
        self.assertTrue(np.all(offset >= -jitter - 1e-12))
        self.assertTrue(np.all(offset <= jitter + 1e-12))
        self.assertTrue(np.all(poses[:, :3] >= env.xyz_bounding_box.low))
        self.assertTrue(np.all(poses[:, :3] <= env.xyz_bounding_box.high))
        # 每一轴都有实际变化，不能只实现 XY 而遗漏插入轴方向。
        self.assertTrue(np.all(np.ptp(offset, axis=0) > 1.8 * jitter))
        reference = Rotation.from_rotvec(env.config.ROTATION_REFERENCE_ROTVEC)
        relative = (Rotation.from_quat(poses[:, 3:]) * reference.inv()).as_rotvec()
        np.testing.assert_allclose(relative, 0.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(poses[:, 3:], axis=1), 1.0, atol=1e-12)
        np.testing.assert_array_equal(env._TARGET_POSE, target_before)

    def test_seed_reproduces_randomization_without_moving_the_nominal_target(self):
        first, second, different = make_env(17), make_env(17), make_env(18)
        first_poses = np.asarray([first.sample_reset_pose() for _ in range(12)])
        second_poses = np.asarray([second.sample_reset_pose() for _ in range(12)])
        other_poses = np.asarray([different.sample_reset_pose() for _ in range(12)])
        np.testing.assert_array_equal(first_poses, second_poses)
        self.assertFalse(np.allclose(first_poses, other_poses))
        np.testing.assert_array_equal(first.config.RESET_POSE, second.config.RESET_POSE)

    def test_slerp_commands_obey_step_caps_and_preserve_unit_quaternions(self):
        env = make_env()
        ref = Rotation.from_rotvec(env.config.ROTATION_REFERENCE_ROTVEC)
        start = env.resetpos.copy()
        start[3:] = (Rotation.from_rotvec(np.deg2rad([-3.0, 1.0, 0.5])) * ref).as_quat()
        goal = start.copy()
        goal[:3] += [0.021, -0.017, -0.012]
        goal[3:] = (Rotation.from_rotvec(np.deg2rad([3.0, -2.0, -0.5])) * ref).as_quat()
        env.actual_pose = start.copy()
        # 很短的请求时长仍必须因步长限制自动分成足够多的命令。
        env.interpolate_move(goal, timeout=0.02)
        poses = np.vstack([start, *[pose for pose, _ in env.commands]])
        steps = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
        self.assertLessEqual(float(np.max(steps)), env.config.MAX_TRANSLATION_STEP_M + 1e-12)
        rotations = Rotation.from_quat(poses[:, 3:])
        angles = (rotations[1:] * rotations[:-1].inv()).magnitude()
        self.assertLessEqual(float(np.max(angles)), env.config.MAX_ROTATION_STEP_RAD + 1e-12)
        np.testing.assert_allclose(np.linalg.norm(poses[:, 3:], axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(poses[-1, :3], goal[:3], atol=1e-12)
        self.assertLess((Rotation.from_quat(poses[-1, 3:]) * Rotation.from_quat(goal[3:]).inv()).magnitude(), 1e-12)
        self.assertTrue(all(source == "reset" for _, source in env.commands))

    def test_reset_withdraws_along_axis_before_lateral_and_rotation_changes(self):
        env = make_env()
        start = env._pose6_to_quat(env._TARGET_POSE)
        outward = -env.config.INSERTION_AXIS_BASE
        start[:3] += 0.01 * outward + np.array([0.003, -0.004, 0.0])
        start[3:] = (Rotation.from_rotvec(np.deg2rad([2.0, 0.0, 0.0])) * Rotation.from_quat(start[3:])).as_quat()
        env.actual_pose = start.copy()
        segments = []
        original = env.interpolate_move

        def record_segment(goal, timeout):
            before = len(env.commands)
            original(goal, timeout)
            segments.append([pose for pose, _ in env.commands[before:]])

        env.interpolate_move = record_segment
        env.go_to_reset()
        self.assertEqual(len(segments), 2)
        retreat = np.asarray(segments[0])
        for pose in retreat:
            difference = pose[:3] - start[:3]
            # 沿退出轴的正交分量应为零；拔出过程中朝向也保持不变。
            np.testing.assert_allclose(difference - np.dot(difference, outward)*outward, 0, atol=1e-12)
            self.assertLess((Rotation.from_quat(pose[3:]) * Rotation.from_quat(start[3:]).inv()).magnitude(), 1e-12)
        distances = (retreat[:, :3] - env._TARGET_POSE[:3]) @ outward
        self.assertTrue(np.all(np.diff(distances) >= -1e-12))
        self.assertGreaterEqual(float(distances[-1]), env.config.RESET_CLEARANCE_M - 1e-12)
        self.assertTrue(np.any(np.abs(np.asarray(segments[1])[:, :2] - start[:2]) > 1e-6))
        np.testing.assert_allclose(env.resetpos, env.actual_pose)

    def test_small_tracking_lag_cannot_start_lateral_motion_before_clearance(self):
        env = make_env()
        start = env._pose6_to_quat(env._TARGET_POSE)
        outward = -env.config.INSERTION_AXIS_BASE
        start[:3] += 0.01 * outward
        env.actual_pose = start.copy()
        original_send = env._send_pos_command
        original_move = env.interpolate_move
        phase = [0]

        def lagged_send(pose, source="reset"):
            original_send(pose, source)
            if phase[0] == 1:
                # 2 mm 小于生产 interpolate_move 的3 mm位置容差。
                env.actual_pose[:3] -= 0.002 * outward

        def inspect_phase(goal, timeout):
            phase[0] += 1
            if phase[0] == 2:
                actual_clearance = np.dot(env.actual_pose[:3] - env._TARGET_POSE[:3], outward)
                self.assertGreaterEqual(actual_clearance, env.config.RESET_CLEARANCE_M - 1e-12,
                                        "Lateral/rotation reset began before actual withdrawal clearance")
            original_move(goal, timeout)

        env._send_pos_command = lagged_send
        env.interpolate_move = inspect_phase
        try:
            env.go_to_reset()
        except RuntimeError:
            # 保持并拒绝开始横移也是合法的安全失败行为。
            self.assertEqual(phase[0], 1)
            self.assertGreater(env.hold_count, 0)

    def test_joint_reset_and_outside_start_are_rejected_before_motion(self):
        env = make_env()
        with self.assertRaisesRegex(ValueError, "Cartesian reset"):
            env.go_to_reset(joint_reset=True)
        self.assertEqual(env.commands, [])
        env.actual_pose[0] = env.xyz_bounding_box.high[0] + 0.01
        with self.assertRaisesRegex(ValueError, "workspace_bounds"):
            env.go_to_reset()
        self.assertEqual(env.commands, [])


class USBResetStopTests(unittest.TestCase):
    """即使配置/视频处理失败，也要先撤销上一回合的运动目标。"""

    def reset_method(self, name):
        tree = ast.parse((ROOT / "examples/experiments/usb_insertion/wrapper.py").read_text(encoding="utf-8-sig"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "USBInsertionEnv")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
        namespace = {"np": np, "LOGGER": logging.getLogger("usb_reset_stop_test")}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "wrapper.py", "exec"), namespace)
        return namespace[name]

    def test_direct_reset_stops_before_feedback_sampling_or_rejection(self):
        method = self.reset_method("go_to_reset")
        for joint_reset in (False, True):
            with self.subTest(joint_reset=joint_reset):
                events = []

                def reject_pose(pose):
                    events.append("validate")
                    raise ValueError("outside calibrated workspace_bounds")

                env = types.SimpleNamespace(
                    hold_position=lambda: events.append("hold"),
                    _update_currpos=lambda: events.append("feedback"),
                    sample_reset_pose=lambda: events.append("sample"),
                    currpos=np.array([0., 0., 0., 0., 0., 0., 1.]),
                    _TARGET_POSE=np.zeros(6),
                    config=types.SimpleNamespace(INSERTION_AXIS_BASE=np.array([0., 0., -1.]), RESET_CLEARANCE_M=0.06),
                    _validate_reset_pose=reject_pose,
                )
                with self.assertRaises(ValueError):
                    method(env, joint_reset=joint_reset)
                self.assertEqual(events, ["hold"] if joint_reset else ["hold", "feedback", "sample", "validate"])

    def test_episode_reset_stops_before_target_video_or_motion_errors(self):
        method = self.reset_method("reset")
        stages = ["target", "video", "go"]
        for failure in stages:
            with self.subTest(failure=failure):
                events = []

                def stage(name):
                    def run(*args, **kwargs):
                        events.append(name)
                        if name == failure:
                            raise RuntimeError("fixture failure at " + name)
                    return run

                env = types.SimpleNamespace(
                    hold_position=lambda: events.append("hold"),
                    select_target_pose=stage("target"),
                    current_target_index=0, current_target_pose=np.zeros(6),
                    save_video=True, save_video_recording=stage("video"),
                    go_to_reset=stage("go"),
                )
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    method(env)
                self.assertEqual(events, ["hold"] + stages[:stages.index(failure) + 1])


class USBConfigGateTests(unittest.TestCase):
    def test_default_uncalibrated_config_rejects_real_environment_before_io(self):
        tree = ast.parse((ROOT / "examples/experiments/usb_insertion/config.py").read_text(encoding="utf-8-sig"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainConfig")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_environment")

        def forbidden_env(*args, **kwargs):
            self.fail("Uncalibrated task must fail before opening devices")
        namespace = {"require_hardware_ready": require_hardware_ready, "USBInsertionEnv": forbidden_env}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "config.py", "exec"), namespace)
        config = types.SimpleNamespace(task_config=load_config())
        with self.assertRaisesRegex(ValueError, "calibration"):
            namespace["get_environment"](config, fake_env=False, classifier=False)


if __name__ == "__main__":
    unittest.main()
