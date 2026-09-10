import copy
import logging
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests
from ur_env.utils.terminal_keyboard import TERMINAL_KEYBOARD_HUB

from ur_env.envs.ur_env import urEnv

LOGGER = logging.getLogger(__name__)


def _fmt_pose(label, value):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return f"{label}={np.array2string(arr, precision=6, separator=', ')}"


class RAMEnv(urEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_regrasp = False
        self._unsubscribe_regrasp = TERMINAL_KEYBOARD_HUB.register(self._on_key)

        LOGGER.info("RAM demo hotkey enabled: press r to regrasp")

    def _debug(self, msg: str):
        # Disabled debug logging to reduce output
        pass

    def _post_debug(self, route: str, payload=None, timeout: float = 5.0):
        url = self.url + route
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            return resp
        except Exception as exc:
            LOGGER.error("[RAMEnv] POST %s failed err=%s", route, exc)
            raise

    def _on_key(self, key_char: str):
        if key_char.lower() == "r":
            self.should_regrasp = True

    def go_to_reset(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """        
        # use compliance mode for coupled reset
        self._update_currpos()
        LOGGER.info("[RAMEnv] go_to_reset start %s", _fmt_pose("currpos", self.currpos))

        self._send_pos_command(self.currpos, source="reset")
        time.sleep(0.3)
        self._post_debug("update_param", payload=self.config.PRECISION_PARAM)

        reset_timeout = float(getattr(self.config, "RESET_INTERPOLATE_TIMEOUT", 2.5))

        # pull up
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        # reset_pose[2] = self.resetpos[2] + 0.04
        reset_pose[2] = self.resetpos[2]
        LOGGER.info(
            "[RAMEnv] pull_up %s %s",
            _fmt_pose("currpos", self.currpos),
            _fmt_pose("pull_up_target", reset_pose),
        )
        self.interpolate_move(reset_pose, timeout=reset_timeout)

        # perform joint reset if needed
        if joint_reset:
            LOGGER.info("[RAMEnv] JOINT RESET requested")
            self._post_debug("jointreset")
            time.sleep(0.5)

        # perform Cartesian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            # Use current orientation as the base, then add a small random rz perturbation.
            rz_noise = np.random.uniform(-self.random_rz_range, self.random_rz_range)
            current_rot = R.from_quat(self.currpos[3:])
            noise_rot = R.from_rotvec(np.array([0.0, 0.0, rz_noise], dtype=np.float64))
            reset_pose[3:] = (noise_rot * current_rot).as_quat()
            LOGGER.info("[RAMEnv] cartesian_reset randomized %s", _fmt_pose("target", reset_pose))
            self.interpolate_move(reset_pose, timeout=reset_timeout)
        else:
            reset_pose = self.resetpos.copy()
            LOGGER.info("[RAMEnv] cartesian_reset fixed %s", _fmt_pose("target", reset_pose))
            self.interpolate_move(reset_pose, timeout=reset_timeout)
        time.sleep(0.5)

        # Change to compliance mode
        self._post_debug("update_param", payload=self.config.COMPLIANCE_PARAM)


    def regrasp(self):
        if not self.config.GRIPPER_ENABLED:
            LOGGER.warning("Regrasp skipped: gripper is disabled for this UR5 setup.")
            return

        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos, source="reset")
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)

        # pull up
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        # reset_pose[2] = self.resetpos[2] + 0.04
        reset_pose[2] = self.resetpos[2] + 0.04
        self.interpolate_move(reset_pose, timeout=1)

        input("Press enter to release gripper...")
        self._send_gripper_command(1.0)
        input("Place RAM in holder and press enter to grasp...")
        top_pose = self.config.GRASP_POSE.copy()
        top_pose[2] += 0.05
        top_pose[0] += np.random.uniform(-0.005, 0.005)
        self.interpolate_move(top_pose, timeout=1)
        time.sleep(0.5)

        grasp_pose = top_pose.copy()
        grasp_pose[2] -= 0.05
        self.interpolate_move(grasp_pose, timeout=0.5)

        requests.post(self.url + "close_gripper_slow")
        self.last_gripper_act = time.time()
        time.sleep(2)

        self.interpolate_move(top_pose, timeout=0.5)
        time.sleep(0.2)

        self.interpolate_move(self.config.RESET_POSE, timeout=1)
        time.sleep(0.5)


    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        # if True:
        if self.should_regrasp:
            self.regrasp()
            self.should_regrasp = False

        self._recover()
        # Respect caller-provided reset mode instead of always forcing a joint reset.
        self.go_to_reset(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self._post_debug("update_param", payload=self.config.COMPLIANCE_PARAM)
        self.terminate = False
        return obs, {}

    def close(self):
        if hasattr(self, "_unsubscribe_regrasp") and self._unsubscribe_regrasp is not None:
            self._unsubscribe_regrasp()
            self._unsubscribe_regrasp = None
        return super().close()
