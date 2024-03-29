#include <cstdio>
#include <ctime>

#include <chrono>
#include <stdexcept>

void nanosleep_chrono(std::chrono::nanoseconds amount) {
  timespec amount_c{
    .tv_sec = amount.count() / std::nano::den,
    .tv_nsec = amount.count() % std::nano::den
  };
  if (nanosleep(&amount_c, nullptr) != 0) {
    throw std::runtime_error("bad sleep");
  }
}

// er... seems like at least for linux, std::this_thread::sleep_for/sleep_until
// uses nanosleep under the hood hehe

int main() {
  using namespace std::literals::chrono_literals;

  nanosleep_chrono(50us);
  nanosleep_chrono(50ms);

  printf("finished sleep\n");

  return 0;
}
