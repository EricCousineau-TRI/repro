#include <sys/prctl.h>

#include <chrono>
#include <thread>
#include <cmath>
#include <Eigen/Dense>
#include <lcm/lcm-cpp.hpp>

#include "drake/lcmt_panda_command.hpp"
#include "drake/lcmt_panda_status.hpp"
#include "drake/common/drake_assert.h"
#include "drake/common/text_logging.h"
#include "timing_profiler/profiler.h"
#include "timing_profiler/profiler_lcm_stats.h"

using drake::lcmt_panda_status;
using drake::lcmt_panda_command;
using lcm::LCM;

namespace libfranka_timing {
namespace {

ProfilerAll& prof_all() {
  static drake::never_destroyed<ProfilerAll> prof_all("PROFILER_DUMB");
  return prof_all.access();
}

void DoMain() {
  // TODO(eric.cousineau): Wasn't really able to figure out how to make this
  // work.
  prctl(PR_SET_TIMERSLACK, 5000U, 0, 0, 0);

  using namespace std::chrono_literals;
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  using clock = std::chrono::steady_clock;
  const auto dt = 1ms;
  // const auto dt = 250us;
  const auto sleep_dt = 10us;
  using Duration = std::chrono::duration<double, std::ratio<1, 1>>;

  LCM lcm("");

  int counter{};
  lcmt_panda_status status{};

  LCM::HandlerFunction<lcmt_panda_status> on_status =
      [&](const lcm::ReceiveBuffer*,
          const std::string&,
          const lcmt_panda_status* message) {
    ++counter;
    status = *message;
  };
  lcm.subscribe("PANDA_STATUS", on_status);

  drake::log()->info("Wait...");
  lcm.handle();
  DRAKE_DEMAND(counter > 0);

  const int ndof = 7;
  Eigen::VectorXd q0(ndof);
  for (int i = 0; i < ndof; ++i) {
    q0(i) = status.joint_position[i];
  }
  Eigen::VectorXd qd(ndof);

  lcmt_panda_command command{};
  command.num_joint_position = ndof;
  command.joint_position.resize(ndof);
  command.control_mode_expected = lcmt_panda_status::CONTROL_MODE_POSITION;

  drake::log()->info("Run");

  const auto t_start = clock::now();
  auto t_current = t_start;
  while (true) {
    static LapTimer& timer_compute = prof_all().profiler().AddTimer("Compute");
    timer_compute.set_strict(false);
    timer_compute.start();

    const auto now = clock::now();
    // double time = Duration(now - t_start).count();  // wall
    double time = Duration(t_current - t_start).count();  // open loop

    command.utime = 1e6 * time;
    if (time >= 5.0) {
      // Hold position.
      time = 5.0;
    }

    double delta_angle = M_PI / 8.0 * (1 - std::cos(M_PI / 2.5 * time));
    qd = q0;
    qd.array() += delta_angle;

    for (int i = 0; i < ndof; ++i) {
      command.joint_position[i] = qd(i);
    }
    lcm.publish("PANDA_COMMAND", &command);

    static LapTimer& timer_across = prof_all().profiler().AddTimer("Across");
    timer_across.lap();

    prof_all().Publish();
    timer_compute.stop();

    static LapTimer& timer_sleep = prof_all().profiler().AddTimer("Sleep");
    timer_sleep.start();
    t_current += dt;
    while (clock::now() < t_current) {
      static LapTimer& timer_sleep_dt =
          prof_all().profiler().AddTimer("SleepDt");
      timer_sleep_dt.start();
      std::this_thread::sleep_for(sleep_dt);
      timer_sleep_dt.stop();
    }
    timer_sleep.stop();
  }

  drake::log()->info("Done");
}

}
}  // namespace anzu

int main() {
  anzu::DoMain();
  return 0;
}
