#include <array>
#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>
#include <gflags/gflags.h>

...

DEFINE_double(
    low_pass_freq, 30.0,
    "Low-pass cutoff frequency (Hz) for desired position or velocity.");
DEFINE_double(
    diff_low_pass_freq, 30.0,
    "Low-pass cutoff frequency (Hz) for computing desired velocities and "
    "accelerations via finite differencing. Note that accelerations are "
    "filtered after being computed from *filtered* velocities.");

DEFINE_bool(
    use_torque_for_position, false,
    R"""(
Use custom torque controller for position control. This uses commanded
position, and then finite differences the position to velocity.

More conceretely, it looks like this:
  torque_command =
      -kp * (q_actual - q_desired) - kd * (v_actual - v_deisred)
      + inertia_matrix(q_actual) * a_desired + coriolis(q_actual, v_actual)
)""");
DEFINE_bool(
    torque_zero_desired_velocity, false,
    "Set desired velocity to zero (do not use finite differencing).");
DEFINE_double(
    torque_coriolis_scale, 1.0,
    "Scaling for Coriolis feedforward term in torque.");
DEFINE_double(
    torque_inertia_scale, 1.0,
    "Scaling for inertia_matrix matrix feedforward term. This uses a finite "
    "differencing of the desired velocity (which seems fine enough for "
    "this control).");
DEFINE_bool(
    torque_feedback_in_acceleration, false,
    "Rather than feedback in torque, meaning gains must account for link "
    "inertias, compute feedback in acceleration, and project through "
    "inertia_matrix matrix (effectively inverse dynamics). This should "
    "have --torque_inerta_scale=1, and replaces the desired acceleration "
    "with this feedback term.");

class PandaDriver {
...
    if (FLAGS_use_torque_for_position) {
      robot_.control(
          std::bind(
              &PandaDriver::DoPositionControlViaTorque,
              this, sp::_1, sp::_2),
          true, franka::kDefaultCutoffFrequency);
    } else {
      robot_.control(
          std::bind(
              &PandaDriver::DoPositionControl,
              this, sp::_1, sp::_2),
          franka::ControllerMode::kJointImpedance,
          true, FLAGS_low_pass_freq);
    }
...

  franka::JointPositions DoPositionControl(
      const franka::RobotState& state, franka::Duration period)  {
    time_integ_ += period.toSec();
    PublishRobotState(state);

    franka::JointPositions q({0, 0, 0, 0, 0, 0, 0});
    for (size_t i = 0; i < q.q.size(); ++i) {
      q.q[i] = q_cmd_latest_(i);
    }

    DRAKE_THROW_UNLESS(
        command_->control_mode_expected ==
        drake::lcmt_panda_status::CONTROL_MODE_POSITION);
    DRAKE_THROW_UNLESS(command_->num_joint_velocity == 0);
    DRAKE_THROW_UNLESS(command_->num_joint_torque == 0);

    if (command_->num_joint_position != state.q.size()) {
      throw std::runtime_error(
          "Received command with unexpected num_joint_position");
    }

    for (int i = 0; i < command_->num_joint_position; ++i) {
      q.q[i] = command_->joint_position[i];
      q_cmd_latest_(i) = q.q[i];
    }

    if (is_first_tick_) {
      is_first_tick_ = false;
    }

    command_prev_ = command_;
    return q;
  }

  franka::Torques DoPositionControlViaTorque(
      const franka::RobotState& state, franka::Duration period)  {
    time_integ_ += period.toSec();
    PublishRobotState(state);

    // Poll for incoming command messages.
    while (lcm_.handleTimeout(0) > 0) {}

    const double dt = period.toSec();
    if (is_first_tick_) {
      // First tick should indicate zero dt.
      DRAKE_DEMAND(dt == 0.0);
      // Initialize command to libfranka's reported last command.
      for (size_t i = 0; i < state.q.size(); ++i) {
        q_cmd_latest_(i) = state.q_d[i];
      }
    } else {
      // N.B. The period from libfranka is in multiples of 1ms.
      DRAKE_DEMAND(dt > 0.0);
    }

    Eigen::VectorXd q_cmd_raw(kNdof);

    DRAKE_THROW_UNLESS(
        command_->control_mode_expected ==
        drake::lcmt_panda_status::CONTROL_MODE_POSITION);
    DRAKE_THROW_UNLESS(command_->num_joint_velocity == 0);
    DRAKE_THROW_UNLESS(command_->num_joint_torque == 0);
    if (command_->num_joint_position != state.q.size()) {
      throw std::runtime_error(
          "Received command with unexpected num_joint_position");
    }
    for (int i = 0; i < command_->num_joint_position; ++i) {
      q_cmd_raw(i) = command_->joint_position[i];
    }

    // Read actual positions and velocities.
    Eigen::VectorXd q_actual(kNdof), v_actual(kNdof);
    for (size_t i = 0; i < kNdof; ++i) {
      q_actual[i] = state.q[i];
      v_actual[i] = state.dq[i];
    }

    // Record previous for finite differencing.
    const Eigen::VectorXd q_cmd_prev = q_cmd_latest_;
    if (is_first_tick_) {
      q_cmd_latest_ = q_cmd_raw;
    } else {
      // Filter commands if enabled.
      if (FLAGS_low_pass_freq < franka::kMaxCutoffFrequency) {
        for (int i = 0; i < kNdof; ++i) {
          q_cmd_latest_[i] = franka::lowpassFilter(
              dt, q_cmd_raw[i], q_cmd_prev[i], FLAGS_low_pass_freq);
        }
      } else {
        q_cmd_latest_ = q_cmd_raw;
      }
    }

    if (is_first_tick_) {
      // Use zero value.
      v_cmd_latest_.setZero();
      a_cmd_latest_.setZero();
    } else {
      // Compute simple finite differencing.
      const Eigen::VectorXd v_cmd_prev = v_cmd_latest_;
      const Eigen::VectorXd v_cmd_raw = (q_cmd_latest_ - q_cmd_prev) / dt;
      for (int i = 0; i < kNdof; ++i) {
        v_cmd_latest_[i] = franka::lowpassFilter(
              dt, v_cmd_raw[i], v_cmd_latest_[i], FLAGS_diff_low_pass_freq);
      }
      const Eigen::VectorXd a_cmd_raw = (v_cmd_latest_ - v_cmd_prev) / dt;
      for (int i = 0; i < kNdof; ++i) {
        a_cmd_latest_[i] = franka::lowpassFilter(
              dt, a_cmd_raw[i], a_cmd_latest_[i], FLAGS_diff_low_pass_freq);
      }
    }

    // Feedback gains.
    Eigen::VectorXd Kp(kNdof), Kd(kNdof);
    Kp.setZero();
    Kd.setZero();
    if (FLAGS_torque_feedback_in_acceleration) {
      // Tested empirically. Still needs work!
      // TODO(eric.cousineau): Do more targeted testing using tools /
      // investigation from prior stuff.
      Kp.setConstant(1000.0);
      Kd.setConstant(45.0);
    } else {
      // Taken originally from libfranka's joint_impedance_control demo, then
      // tweaked.
      // TODO(eric.cousineau): Refine further using tools / investigation from
      // prior stuff.
      Kp << 875.0, 1050.0, 1050.0, 875.0, 175.0, 350.0, 87.5;
      Kd << 37.5, 50.0, 37.5, 25.0, 5.0, 3.75, 2.5;
    }

    // Feedback term.
    Eigen::VectorXd v_desired = v_cmd_latest_;
    if (FLAGS_torque_zero_desired_velocity) {
      v_desired.setZero();
    }
    Eigen::VectorXd feedback =
        -Kp.array() * (q_actual - q_cmd_latest_).array()
        - Kd.array() * (v_actual - v_desired).array();

    // Compute Coriolis and inerta terms from Franka model.
    const std::array<double, kNdof> coriolis_array = model_.coriolis(state);
    const Eigen::VectorXd coriolis_vector =
        Eigen::VectorXd::Map(&coriolis_array[0], kNdof)
        * FLAGS_torque_coriolis_scale;
    const std::array<double, kNdof * kNdof> inertia_array = model_.mass(state);
    const Eigen::MatrixXd inertia_matrix =
        Eigen::MatrixXd::Map(&inertia_array[0], kNdof, kNdof)
        * FLAGS_torque_inertia_scale;

    // N.B. Franka docs say that `franka::Torques` should indicate desired
    // torques without gravity or friction.
    Eigen::VectorXd tau_cmd(kNdof);
    if (FLAGS_torque_feedback_in_acceleration) {
      // Project feedback (in acceleration) to torques, add feedforward
      // coriolis.
      // TODO(eric.cousineau): Still provide option to add feedfoward inertia
      // terms?
      tau_cmd = inertia_matrix * feedback + coriolis_vector;
    } else {
      // Use torques directly, add feedforward based on command acceleration.
      tau_cmd = feedback + inertia_matrix * a_cmd_latest_ + coriolis_vector;
    }

    if (is_first_tick_) {
      is_first_tick_ = false;
    }

    std::array<double, kNdof> tau_cmd_array{};
    Eigen::VectorXd::Map(&tau_cmd_array[0], kNdof) = tau_cmd;
    return franka::Torques(tau_cmd_array);
  }
...
};
