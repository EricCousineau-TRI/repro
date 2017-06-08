#include <chrono>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>
#include <thread>

// @ref https://stackoverflow.com/questions/1300180/ignore-openmp-on-machine-that-doesnt-have-it
#ifdef _OPENMP
#include <omp.h>
#endif

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

class ScopedTimer {
 public:
  ScopedTimer()
      : start_(Clock::now()) {}
  ~ScopedTimer() {
    cout << "Elapsed time: " << Duration(Clock::now() - start_).count() << endl;
  }
 private:
  TimePoint start_;
};
auto start = Clock::now();

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
  
  if (!no_pragma) {
    // Do a warm-up to see if we can get the threads started
    {
      ScopedTimer timer;
      #pragma omp parallel for
      for (int i = 0; i < count; ++i) {
        value[i] = 0.;
      }
    }
    {
      ScopedTimer timer;
      #pragma omp parallel for
      for (int i = 0; i < count; ++i) {
        #ifdef _OPENMP
        std::ostringstream os;
        os << "thread: " << omp_get_thread_num() << endl;
        cout << os.str();
        #endif
        value[i] = expensive(i, no_sleep);
      }
    }
  } else {
    ScopedTimer timer;
    for (int i = 0; i < count; ++i) {
      value[i] = expensive(i, no_sleep);
    }
  }
  
  return 0;
}
