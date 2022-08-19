/*
Copyright (c) 2017 Franka Emika GmbH (for original code)
Use of this source code is governed by the Apache-2.0 license, see LICENSE

Modified to show issues with different time sources.
*/

#include <cmath>
#include <chrono>
#include <iostream>
#include <sstream>

#include <franka/exception.h>
#include <franka/robot.h>

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>;

template <typename T>
T FromString(const std::string& s) {
  T out{};
  std::istringstream(s) >> out;
  return out;
}

enum class TimeMode {
  Status = 0,
  StatusDelayed = 1,
  Wall = 2,
  OpenLoop = 3,
};

template <>
TimeMode FromString(const std::string& s) {
  if (s == "Status") {
    return TimeMode::Status;
  } else if (s == "StatusDelayed") {
    return TimeMode::StatusDelayed;
  } else if (s == "Wall") {
    return TimeMode::Wall;
  } else if (s == "OpenLoop") {
    return TimeMode::OpenLoop;
  } else {
    std::abort();
  }
}

constexpr double kDesiredTimeStep = 0.001;

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr
        << "Usage: " << argv[0]
        << " <robot-hostname> <time_mode> <cutoff_freq> <gain_scale>"
        << std::endl;
    return -1;
  }
  try {
    franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
    const TimeMode time_mode = FromString<TimeMode>(argv[2]);
    const double cutoff_freq = FromString<double>(argv[3]);
    const double gain_scale = FromString<double>(argv[4]);

    std::array<double, 7> K_theta{
        {3000, 3000, 3000, 2500, 2500, 2000, 2000}};
    for (int i = 0; i < 7; ++i) {
      K_theta[i] *= gain_scale;
    }
    robot.setJointImpedance(K_theta);

    std::array<double, 7> initial_position;
    double time = 0.0;
    const double total_time = 5.0;

    auto update_delta = [&](
        const franka::RobotState& robot_state, franka::Duration period) {
      const auto now = Clock::now();
      static TimePoint start = now;

      if (time_mode == TimeMode::Status) {
        // Smooth.
        time += period.toSec();
      } else if (time_mode == TimeMode::Wall) {
        // WARNING: Crackly, may fault.
        time = Duration(now - start).count();
      }

      if (time == 0.0) {
        initial_position = robot_state.q_d;
      }

      const double max_delta_angle = M_PI / 8.0;

      const double omega = 2 * M_PI / total_time;
      double delta_angle = max_delta_angle * (1 - std::cos(omega * time));

      if (time_mode == TimeMode::StatusDelayed) {
        // Slight crackles at reported delay points.
        time += period.toSec();
      } else if (time_mode == TimeMode::OpenLoop) {
        // WARNING: Bad! Will hit huge delay.
        time += kDesiredTimeStep;
      }

      return delta_angle;
    };

    auto callback = [&](
        const franka::RobotState& robot_state, franka::Duration period)
        -> franka::JointPositions {
      const double delta_angle = update_delta(robot_state, period);
      franka::JointPositions output = {
          {initial_position[0] + delta_angle,
           initial_position[1] + delta_angle,
           initial_position[2] + delta_angle,
           initial_position[3] + delta_angle,
           initial_position[4] + delta_angle,
           initial_position[5] + delta_angle,
           initial_position[6] + delta_angle}};
      if (time >= total_time) {
        return franka::MotionFinished(output);
      }
      return output;
    };

    const bool limit_rate = true;
    robot.control(
        callback, franka::ControllerMode::kJointImpedance,
        limit_rate, cutoff_freq);
  } catch (const franka::Exception& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;
}
