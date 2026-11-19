"""JSON 的离线防错检查，不导入相机或机器人运行环境。"""
import copy
import unittest

from examples.experiments.usb_insertion.runtime_config import load_config, validate_config, require_hardware_ready, controller_safety_payload
from examples.check_usb_training_setup import check_setup


class USBRuntimeConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_default_is_valid_but_blocks_real_motion(self):
        report = check_setup()
        self.assertTrue(report["schema_valid"])
        self.assertFalse(report["ready"])
        self.assertEqual(report["action_dim"], 6)
        with self.assertRaisesRegex(ValueError, "calibration"):
            require_hardware_ready(self.config)

    def test_full_reset_range_not_only_center_is_checked(self):
        config = copy.deepcopy(self.config)
        config["reset"]["translation_range_m"][0] = 0.05
        with self.assertRaisesRegex(ValueError, "complete reset"):
            validate_config(config)

    def test_no_gripper_or_network_dimension_mismatch(self):
        for key, value in (("network_action_dim", 7), ("network_action_dim", 3)):
            config = copy.deepcopy(self.config)
            config["training"][key] = value
            with self.assertRaisesRegex(ValueError, "action_dim"):
                validate_config(config)
        config = copy.deepcopy(self.config)
        config["observation"]["proprio_keys"].append("gripper_pose")
        with self.assertRaisesRegex(ValueError, "gripper_pose"):
            validate_config(config)

    def test_motion_limits_and_randomization_reject_nan_and_excess(self):
        for value in (float("nan"), 0.0, 0.021):
            config = copy.deepcopy(self.config)
            config["action"]["max_translation_step_m"] = value
            with self.assertRaises(ValueError):
                validate_config(config)
        config = copy.deepcopy(self.config)
        config["reset"]["rotation_range_deg"][0] = 6.0
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_camera_and_clearance_must_cover_randomization(self):
        config = copy.deepcopy(self.config)
        config["observation"]["cameras"]["wrist"]["crop_yxyx"] = [0, 1000, 0, 2000]
        with self.assertRaisesRegex(ValueError, "crop"):
            validate_config(config)
        config = copy.deepcopy(self.config)
        config["reset"]["clearance_along_axis_m"] = 0.09
        with self.assertRaisesRegex(ValueError, "clearance"):
            validate_config(config)

    def test_calibration_flags_do_not_bypass_missing_force_limits(self):
        config = copy.deepcopy(self.config)
        config["controller_safety"]["wrench_monitor_enabled"] = True
        config["calibration"] = {name: True for name in config["calibration"]}
        config["calibration"]["status"] = "calibrated"
        with self.assertRaisesRegex(ValueError, "force_norm_limit"):
            require_hardware_ready(config)
        config["controller_safety"].update(force_norm_limit_n=5.0, torque_norm_limit_nm=0.5)
        require_hardware_ready(config)
        payload = controller_safety_payload(config)
        self.assertEqual(payload["control_hz"], 50)
        self.assertEqual(payload["workspace_low"], config["workspace_bounds"]["position_low_m"])
        self.assertAlmostEqual(payload["rotation_limits_rad"][0], 0.0872664626)

    def test_position_only_ur5_does_not_require_force_sensor(self):
        config = copy.deepcopy(self.config)
        for name in ("tcp_verified", "workspace_verified", "reset_path_verified", "camera_fov_verified"):
            config["calibration"][name] = True
        config["calibration"]["status"] = "calibrated"
        self.assertFalse(config["controller_safety"]["wrench_monitor_enabled"])
        self.assertIsNone(config["controller_safety"]["force_norm_limit_n"])
        require_hardware_ready(config)


if __name__ == "__main__":
    unittest.main()
