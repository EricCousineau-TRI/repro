#include <chrono>
#include <cmath>
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

double expensive(int i) {
  sleep(0.05);
  return i * i;
}

int main() {
  const int count = 100;
  double value[count];
  auto start = Clock::now();
  #pragma omp parallel for
  for (int i = 0; i < count; ++i) {
    value[i] = expensive(i);
  }
  cout << "Elapsed time: " << Duration(Clock::now() - start).count() << endl;
  return 0;
}
