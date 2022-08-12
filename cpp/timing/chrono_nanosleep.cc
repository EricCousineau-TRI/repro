#include <cstdio>
#include <ctime>

#include <chrono>
#include <stdexcept>

void sleep_high_resolution(std::chrono::nanoseconds amount) {
  timespec amount_c{
    .tv_sec = amount.count() / std::nano::den,
    .tv_nsec = amount.count() % std::nano::den
  };
  if (nanosleep(&amount_c, nullptr) != 0) {
    throw std::runtime_error("bad sleep");
  }
}

int main() {
  using namespace std::literals::chrono_literals;

  sleep_high_resolution(50us);
  sleep_high_resolution(50ms);

  printf("finished sleep\n");

  return 0;
}
