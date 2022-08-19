# Utilities for libfranka Timing

Basic export from anzu; does not compile yet.

relates:
- https://github.com/frankaemika/libfranka/issues/120
- https://github.com/EricCousineau-TRI/franka_panda_udp_timing/

```sh
# experiment: time sources
./run //:panda_generate_joint_position_motion <fci-ip> Status 100 1.0
./run //:panda_generate_joint_position_motion <fci-ip> StatusDelayed 100 1.0
./run //:panda_generate_joint_position_motion <fci-ip> Wall 30 1.0
./run //:panda_generate_joint_position_motion <fci-ip> OpenLoop 30 1.0
# - sub-experiment: using lcm
# - - terminal 1
./run <driver> --low_pass_freq=30.0  
# - - terminal 2
./run //:panda_controller_lcm

# experiment: gains / feedforward
./run //:panda_generate_joint_position_motion <fci-ip> Wall 30 1
./run //:panda_generate_joint_position_motion <fci-ip> Wall 30 0.1
./run //:panda_generate_joint_position_motion <fci-ip> Wall 30 0.01
./run //:panda_generate_joint_position_motion <fci-ip> Wall 30 0.0
```

Consider using `chrt -r 20` on procs. See `../soft_real_time/README.md`.
