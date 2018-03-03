#include <chrono>
#include <iostream>
#include <thread>

int main() {
  std::thread daemon([]() {
    while (true) {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(100ms);
      std::cerr << "Tick\n";
    }
  });
  daemon.detach();

  std::cout << "Finish\n";

  return 0;
}
