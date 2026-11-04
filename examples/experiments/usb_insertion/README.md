# USB insertion

This task uses the same UR5 reset and streamed pose-servo flow as
`ram_insertion`. The plug orientation and insertion position are fixed to one
calibrated pose.

Edit `EnvConfig.BASE_TARGET_POSE` after calibrating the USB socket. The target
API still exposes `TARGET_POSES`, but it contains exactly one pose generated
from the zero translation offset. `USBInsertionEnv` therefore selects index 0
on every reset, and no 3x3 target grid or target randomization is used.

The reset pose is fixed as well (`RANDOM_RESET = False`). If the calibrated
pose or reset offset changes, check `ABS_POSE_LIMIT_LOW/HIGH` before enabling
the real robot.
