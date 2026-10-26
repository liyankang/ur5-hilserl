# USB insertion

This task uses the same UR5 reset and streamed pose-servo flow as `ram_insertion`. The
plug orientation is fixed and the insertion point is translated in the XY
plane.

Edit `EnvConfig.BASE_TARGET_POSE` after calibrating the USB socket. Candidate
positions are defined in metres by `TARGET_TRANSLATION_OFFSETS`; the default is
a 3x3 grid with 20 mm spacing. `TARGET_POSES` is generated automatically and
all candidates keep the same final three rotation values.

At every reset, `USBInsertionEnv` selects one candidate. Training uses random
selection by default. Set `TARGET_SELECTION = "cycle"` for balanced episodes,
or call `env.reset(options={"target_index": 4})` to evaluate one position.

The safety box includes all candidate positions and the XY reset noise. If the
grid or reset range changes, check `ABS_POSE_LIMIT_LOW/HIGH` before enabling
the real robot. Positions outside the candidate set are not guaranteed to
work; collect demonstrations across the intended range and test intermediate
positions separately.
