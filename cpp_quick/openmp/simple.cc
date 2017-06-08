#include <chrono>
#include <iostream>

using namespace std;

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<double>;
using TimePoint = std::chrono::time_point<Clock, Duration>;

int main() {
  const int count = 1000000;
  double value[count];
  auto start = Clock::now();
  #pragma omp parallel for
  for (int i = 0; i < count; ++i) {
    value[i] = i*i;
  }
  cout << "Elapsed time: " << Duration(Clock::now() - start).count() << endl;
  return 0;
}
