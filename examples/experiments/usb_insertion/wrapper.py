"""USB insertion environment with per-episode translated insertion targets."""

import logging

import numpy as np

from experiments.ram_insertion.wrapper import RAMEnv

LOGGER = logging.getLogger(__name__)


class USBInsertionEnv(RAMEnv):
    """RAM-style UR5 environment for a fixed-tool USB plug.

    The plug orientation is fixed.  At reset, one candidate target is chosen
    and all target-dependent state in ``urEnv`` is updated before the reset
    motion starts.  This keeps reward evaluation and safety clipping aligned
    with the physical insertion point.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_target_index = 0
        self.current_target_id = 0
        self.current_target_pose = np.asarray(self._TARGET_POSE, dtype=np.float64).copy()
        self._episode_count = 0
        self._set_target_pose(self.current_target_pose)

    @property
    def target_pose(self):
        """Return the active target pose without exposing mutable state."""
        return self.current_target_pose.copy()

    def _sample_target_index(self, target_index=None):
        targets = np.asarray(self.config.TARGET_POSES)
        if targets.ndim != 2 or targets.shape[1] != 6 or len(targets) == 0:
            raise ValueError("config.TARGET_POSES must have shape (N, 6) with N > 0")

        if target_index is not None:
            index = int(target_index)
            if index < 0 or index >= len(targets):
                raise IndexError(
                    f"target_index {index} is outside [0, {len(targets) - 1}]"
                )
            return index

        selection = getattr(self.config, "TARGET_SELECTION", "random")
        if selection == "cycle":
            return self._episode_count % len(targets)
        if selection != "random":
            raise ValueError(
                f"Unsupported TARGET_SELECTION={selection!r}; use 'random' or 'cycle'"
            )
        return int(np.random.randint(len(targets)))

    def _set_target_pose(self, target_pose):
        """Synchronize the active target, reset pose, and reward target."""
        target_pose = np.asarray(target_pose, dtype=np.float64).reshape(-1)
        if target_pose.shape != (6,):
            raise ValueError(f"target_pose must have shape (6,), got {target_pose.shape}")
        base_orientation = np.asarray(self.config.BASE_TARGET_POSE, dtype=np.float64)[3:]
        if not np.allclose(target_pose[3:], base_orientation):
            raise ValueError(
                "USB insertion targets may translate the plug, but may not "
                "change its orientation"
            )

        # The final three components are copied from the configured candidate
        # pose; this method never introduces a rotation perturbation.
        self.current_target_pose = target_pose.copy()
        self._TARGET_POSE = target_pose.copy()
        self.config.TARGET_POSE = target_pose.copy()
        self.config.GRASP_POSE = target_pose.copy()

        reset_offset = np.asarray(
            getattr(self.config, "RESET_TARGET_OFFSET", [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        if reset_offset.shape != (6,):
            raise ValueError("RESET_TARGET_OFFSET must have shape (6,)")
        reset_pose = target_pose + reset_offset
        self._RESET_POSE = reset_pose.copy()
        self.config.RESET_POSE = reset_pose.copy()
        self.resetpos = self._pose6_to_quat(reset_pose)

        # The static safety box is built to contain every candidate target and
        # its reset perturbation.  Fail early if a custom target set violates it.
        target_in_box = np.all(target_pose[:3] >= self.xyz_bounding_box.low) and np.all(
            target_pose[:3] <= self.xyz_bounding_box.high
        )
        reset_in_box = np.all(reset_pose[:3] >= self.xyz_bounding_box.low) and np.all(
            reset_pose[:3] <= self.xyz_bounding_box.high
        )
        if not target_in_box or not reset_in_box:
            raise ValueError(
                "Active USB target or reset pose is outside "
                f"ABS_POSE_LIMIT_LOW/HIGH: target={target_pose[:3]}, "
                f"reset={reset_pose[:3]}"
            )

    def select_target_pose(self, target_index=None):
        """Select and activate a target without starting a reset motion."""
        index = self._sample_target_index(target_index)
        self.current_target_index = index
        self.current_target_id = index
        self._set_target_pose(np.asarray(self.config.TARGET_POSES[index]))
        self._episode_count += 1
        return self.target_pose

    def reset(self, joint_reset=False, target_index=None, options=None, **kwargs):
        """Select a target, then run the RAM-compatible reset sequence.

        ``target_index`` is useful for evaluation and calibration.  Training
        can omit it to sample uniformly, or set ``TARGET_SELECTION='cycle'``
        to guarantee balanced coverage across episodes.
        """
        if target_index is None and options is not None:
            target_index = options.get("target_index")

        self.select_target_pose(target_index)
        LOGGER.info(
            "USB target index=%d pose=%s",
            self.current_target_index,
            np.array2string(self.current_target_pose, precision=5),
        )
        return super().reset(joint_reset=joint_reset, options=options, **kwargs)


# Short alias matching the existing USB task naming in older experiments.
USBEnv = USBInsertionEnv

