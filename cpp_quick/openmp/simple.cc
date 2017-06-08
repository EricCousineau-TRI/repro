#include <chrono>
#include <cmath>
#include <string>
#include <iostream>
#include <thread>

using namespace std;

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<double>;
using TimePoint = std::chrono::time_point<Clock, Duration>;

inline void sleep(double seconds) {
  int ms = std::round(seconds * 1000);
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

double expensive(int i, bool no_sleep) {
  if (!no_sleep) {
    sleep(0.05);
  }
  return i * i;
}

int main(int argc, char** argv) {
  bool no_sleep = false;
  bool no_pragma = false;
  {
    int i = 1;
    while (i < argc) {
      string arg = argv[i];
      if (arg == "--no-sleep") {
        no_sleep = true;
      } else if (arg == "--no-pragma") {
        no_pragma = true;
      } else {
        cerr << "usage:  " << argv[0] << " [--no-sleep] [--no-pragma]" << endl;
        return 1;
      }
      i++;
    }
  }
  const int count = 10;
  double value[count];
  auto start = Clock::now();
  if (!no_pragma) {
    #pragma omp parallel for
    for (int i = 0; i < count; ++i) {
      value[i] = expensive(i, no_sleep);
    }
  } else {
    for (int i = 0; i < count; ++i) {
      value[i] = expensive(i, no_sleep);
    }
  }
  cout << "Elapsed time: " << Duration(Clock::now() - start).count() << endl;
  return 0;
}
