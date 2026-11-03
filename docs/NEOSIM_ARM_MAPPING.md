# NeoSim explicit arm mappings

Single-arm tasks default to `embodiment`. Dual-arm HDF5 requires explicit
`robot.arms=2` and `robot.embodiment_paths=["embodiment","embodiment_b"]`.
Each arm contributes position, column rot6d and normalized opening (10 channels).
Next-observation targets shift all arms together by one recorded sample.

For the inspected ARX5 memory capture, each joint stream has eight columns;
finger indices are [6,7], physical total stroke 0.078 m, simulation clock 120 Hz,
recording stride four (30 Hz). There are head, wrist and wrist_b RGB streams and
four tactile streams. Other embodiments retain their own explicit clock, stroke
and finger mapping; do not reuse the ARX5 values for Franka.

Joint prepared-task configurations propagate each child's camera keys, tactile
keys, action channels and normalization stats separately. Task-specific topology
must be preserved in the latent pool.
