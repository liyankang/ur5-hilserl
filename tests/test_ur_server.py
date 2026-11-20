import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import werkzeug

from serl_robot_ur.robot_servers.ur_server import URStreamingController, create_app, ensure_quat_pose

if not hasattr(werkzeug, "__version__"):
    werkzeug.__version__ = "test"


class FakeArmBackend:
    def __init__(self, robot_ip, reset_joint_target):
        self.robot_ip = robot_ip
        self.reset_joint_target = list(reset_joint_target)
        self.pose = [0.4, -0.1, 0.2, 0.0, 0.0, 0.0, 1.0]
        self.vel = [0.0] * 6
        self.force = [0.0, 0.0, 0.0]
        self.torque = [0.0, 0.0, 0.0]
        self.q = [0.0] * 6
        self.dq = [0.0] * 6
        self.jacobian = [[0.0] * 6 for _ in range(6)]
        self.last_pose_command = None
        self.last_movej_command = None
        self.joint_reset_targets = []
        self.updated_params = []
        self.payloads = []
        self.force_commands = []
        self.force_stop_calls = 0
        self.hold_calls = 0
        self.clear_calls = 0
        self.closed = False

    def close(self):
        self.closed = True

    def set_target_pose(self, pose, speed=None, accel=None, source="policy"):
        pose = ensure_quat_pose(pose)
        self.pose = pose.tolist()
        self.last_pose_command = {
            "pose": self.pose,
            "speed": speed,
            "accel": accel,
            "source": source,
        }
        return self.pose

    def move_joints(self, q, speed=None, accel=None):
        self.q = list(q)
        self.last_movej_command = {
            "q": self.q,
            "speed": speed,
            "accel": accel,
        }
        return self.q

    def joint_reset(self, target=None, speed=None, accel=None):
        target = self.reset_joint_target if target is None else list(target)
        self.joint_reset_targets.append(
            {
                "target": list(target),
                "speed": speed,
                "accel": accel,
            }
        )
        self.q = list(target)
        return target

    def get_state(self):
        return {
            "pose": list(self.pose),
            "vel": list(self.vel),
            "force": list(self.force),
            "torque": list(self.torque),
            "q": list(self.q),
            "dq": list(self.dq),
            "jacobian": [list(row) for row in self.jacobian],
            "gripper_pos": [0.0],
        }

    def update_param(self, params):
        self.updated_params.append(dict(params))
        return {"applied": {"translational_stiffness": params.get("translational_stiffness")}, "ignored": []}

    def set_payload(self, payload):
        self.payloads.append(dict(payload))
        return {"applied": True}

    def clear_error(self):
        self.clear_calls += 1
        return "Cleared"

    def set_force_mode(self, command):
        self.force_commands.append(dict(command))
        return dict(command)

    def stop_force_mode(self):
        self.force_stop_calls += 1

    def hold(self):
        self.hold_calls += 1


class URArmControlRoutesTest(unittest.TestCase):
    def setUp(self):
        def backend_factory(**kwargs):
            self.backend = FakeArmBackend(**kwargs)
            return self.backend

        app = create_app(
            robot_ip="192.168.25.18",
            gripper_type="None",
            reset_joint_target=[-1.34, -1.58, -1.9, -1.1, 1.58, 1.81],
            backend_factory=backend_factory,
        )
        self.client = app.test_client()

    def test_pose_movej_and_getstate_cover_arm_control(self):
        pose = [0.5, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        pose_response = self.client.post("/pose", json={"arr": pose, "speed": 0.2, "accel": 0.3})
        self.assertEqual(pose_response.status_code, 200)
        self.assertEqual(self.backend.last_pose_command["pose"], pose)
        self.assertEqual(self.backend.last_pose_command["speed"], 0.2)
        self.assertEqual(self.backend.last_pose_command["accel"], 0.3)
        self.assertEqual(self.backend.last_pose_command["source"], "policy")

        movej = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
        movej_response = self.client.post("/movej", json={"q": movej, "speed": 0.4, "accel": 0.5})
        self.assertEqual(movej_response.status_code, 200)
        self.assertEqual(self.backend.last_movej_command["q"], movej)
        self.assertEqual(self.backend.last_movej_command["speed"], 0.4)
        self.assertEqual(self.backend.last_movej_command["accel"], 0.5)

        state_response = self.client.post("/getstate")
        self.assertEqual(state_response.status_code, 200)
        state = state_response.get_json()

        self.assertEqual(len(state["pose"]), 7)
        self.assertEqual(len(state["vel"]), 6)
        self.assertEqual(len(state["force"]), 3)
        self.assertEqual(len(state["torque"]), 3)
        self.assertEqual(len(state["q"]), 6)
        self.assertEqual(len(state["dq"]), 6)
        self.assertEqual(len(state["jacobian"]), 6)
        self.assertEqual(len(state["jacobian"][0]), 6)
        self.assertEqual(len(state["gripper_pos"]), 1)

    def test_jointreset_and_clearerr_drive_arm_backend(self):
        reset_response = self.client.post("/jointreset")
        self.assertEqual(reset_response.status_code, 200)
        self.assertEqual(self.backend.joint_reset_targets[-1]["target"], self.backend.reset_joint_target)

        custom_target = [0.2, -1.2, 1.1, -0.7, 1.0, 0.15]
        custom_reset = self.client.post("/jointreset", json={"target": custom_target, "speed": 0.9, "accel": 0.7})
        self.assertEqual(custom_reset.status_code, 200)
        self.assertEqual(self.backend.joint_reset_targets[-1]["target"], custom_target)
        self.assertEqual(self.backend.joint_reset_targets[-1]["speed"], 0.9)
        self.assertEqual(self.backend.joint_reset_targets[-1]["accel"], 0.7)

        clear_response = self.client.post("/clearerr")
        self.assertEqual(clear_response.status_code, 200)
        self.assertEqual(self.backend.clear_calls, 1)

    def test_pose_route_accepts_rotvec_pose_for_arm_motion(self):
        rotvec_pose = [0.45, -0.15, 0.25, 0.0, 0.0, 1.57079632679]
        response = self.client.post("/pose", json={"arr": rotvec_pose})
        self.assertEqual(response.status_code, 200)

        target_pose = self.backend.last_pose_command["pose"]
        self.assertEqual(len(target_pose), 7)
        self.assertAlmostEqual(target_pose[0], 0.45)
        self.assertAlmostEqual(target_pose[1], -0.15)
        self.assertAlmostEqual(target_pose[2], 0.25)
        self.assertAlmostEqual(sum(x * x for x in target_pose[3:]) ** 0.5, 1.0, places=6)

    def test_force_mode_routes_start_and_stop_direct_force_control(self):
        command = {"wrench": [0, 0, 3, 0, 0, 0], "selection_vector": [0, 0, 1, 0, 0, 0]}
        response = self.client.post("/force_mode", json=command)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.backend.force_commands, [command])
        self.assertEqual(response.get_json()["status"], "active")

        stop_response = self.client.post("/force_mode_stop")
        self.assertEqual(stop_response.status_code, 200)
        self.assertEqual(self.backend.force_stop_calls, 1)

    def test_hold_reports_success_only_when_backend_stop_succeeds(self):
        response = self.client.post("/hold")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "held"})
        self.assertEqual(self.backend.hold_calls, 1)

        with patch.object(self.backend, "hold", side_effect=RuntimeError("RTDE did not confirm forceModeStop")):
            response = self.client.post("/hold")
        self.assertEqual(response.status_code, 503)
        self.assertIn("forceModeStop", response.get_json()["error"])


class URStreamingControllerHoldTest(unittest.TestCase):
    pose = [0.5, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0]

    def make_controller(self, rtde_control, mode="forcemode"):
        with patch("threading.Thread.start"):
            return URStreamingController(
                robot_ip="unused",
                reset_joint_target=[0.0] * 6,
                controller_mode=mode,
                rtde_control_cls=lambda _: rtde_control,
                rtde_receive_cls=lambda _: SimpleNamespace(),
            )

    def test_hold_clears_force_command_and_resumes_on_a_fresh_policy_pose(self):
        force_stop = Mock(return_value=True)
        controller = self.make_controller(SimpleNamespace(forceModeStop=force_stop))
        controller._force_mode_active = True
        controller.set_force_mode({"wrench": [0, 0, 1, 0, 0, 0]})

        controller.hold()

        force_stop.assert_called_once_with()
        self.assertTrue(controller._servo_paused)
        self.assertFalse(controller._force_mode_active)
        self.assertIsNone(controller._direct_force_command)
        self.assertEqual(controller.last_policy_target_time, 0.0)

        controller.set_target_pose(self.pose, source="reset")
        self.assertTrue(controller._servo_paused)
        with self.assertRaisesRegex(ValueError, "finite"):
            controller.set_target_pose([float("nan"), *self.pose[1:]], source="policy")
        with self.assertRaisesRegex(ValueError, "finite"):
            controller.set_target_pose(self.pose, speed=float("inf"), source="policy")
        self.assertTrue(controller._servo_paused)

        controller.set_target_pose(self.pose, source="policy")
        self.assertFalse(controller._servo_paused)
        self.assertGreater(controller.last_policy_target_time, 0.0)

    def test_failed_force_stop_leaves_motion_paused(self):
        force_stop = Mock(return_value=False)
        servo_stop = Mock(return_value=True)
        controller = self.make_controller(SimpleNamespace(forceModeStop=force_stop, servoStop=servo_stop))

        with self.assertRaisesRegex(RuntimeError, "forceModeStop"):
            controller.hold()

        force_stop.assert_called_once_with()
        servo_stop.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "motion hold"):
            controller.set_target_pose(self.pose, source="policy")
        self.assertTrue(controller._servo_paused)

    def test_missing_servo_stop_does_not_confirm_hold(self):
        controller = self.make_controller(SimpleNamespace(), mode="servo")

        with self.assertRaisesRegex(RuntimeError, "servoStop"):
            controller.hold()

        self.assertTrue(controller._servo_paused)

    def test_joint_move_does_not_resume_held_streaming(self):
        move_j = Mock(return_value=True)
        servo_stop = Mock(return_value=True)
        controller = self.make_controller(SimpleNamespace(moveJ=move_j, servoStop=servo_stop), mode="servo")

        controller.hold()
        controller.move_joints([0.0] * 6)

        move_j.assert_called_once()
        self.assertTrue(controller._servo_paused)

    def test_pose_request_already_processing_cannot_resume_after_hold(self):
        controller = self.make_controller(SimpleNamespace(servoStop=lambda: True), mode="servo")

        def hold_before_target_update(pose):
            controller.hold()
            return ensure_quat_pose(pose)

        with patch("serl_robot_ur.robot_servers.ur_server.ensure_quat_pose", side_effect=hold_before_target_update):
            with self.assertRaisesRegex(RuntimeError, "motion hold"):
                controller.set_target_pose(self.pose, source="policy")

        self.assertTrue(controller._servo_paused)

    def test_control_loop_checks_hold_before_dispatch(self):
        servo_l = Mock(return_value=True)
        servo_stop = Mock(return_value=True)
        controller = self.make_controller(SimpleNamespace(servoL=servo_l, servoStop=servo_stop), mode="servo")

        def hold_during_safety_check():
            controller.hold()
            controller._stop_event.set()
            return True

        with patch.object(controller, "_check_safety", side_effect=hold_during_safety_check):
            controller._run()

        servo_stop.assert_called_once_with()
        servo_l.assert_not_called()

    def test_control_loop_discards_old_target_when_hold_resumes_before_dispatch(self):
        servo_l = Mock(return_value=True)
        controller = self.make_controller(SimpleNamespace(servoL=servo_l, servoStop=lambda: True), mode="servo")
        controller.set_target_pose([0.1, *self.pose[1:]])

        def hold_and_resume_during_safety_check():
            controller._stop_event.set()
            controller.hold()
            controller.set_target_pose(self.pose)
            return True

        with patch.object(controller, "_check_safety", side_effect=hold_and_resume_during_safety_check):
            controller._run()

        servo_l.assert_called_once()
        self.assertEqual(servo_l.call_args.args[0][:3], self.pose[:3])


if __name__ == "__main__":
    unittest.main()
