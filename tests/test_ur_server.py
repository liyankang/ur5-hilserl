import unittest

import werkzeug

from serl_robot_ur.robot_servers.ur_server import create_app, ensure_quat_pose

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
        self.clear_calls = 0
        self.closed = False

    def close(self):
        self.closed = True

    def set_target_pose(self, pose, speed=None, accel=None):
        pose = ensure_quat_pose(pose)
        self.pose = pose.tolist()
        self.last_pose_command = {
            "pose": self.pose,
            "speed": speed,
            "accel": accel,
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


if __name__ == "__main__":
    unittest.main()
