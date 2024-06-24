#undef NDEBUG

#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstdlib>

#include <thread>
#include <vector>

volatile bool ready{false};
volatile bool done{false};

void on_sigint(int) {
  done = true;
}

void busy_wait() {
  printf("Start\n");
  while (!ready) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  volatile int count = 0;
  while (!done) {
    ++count;
  }
  printf("Finish\n");
}

int main(int argc, char** argv) {
  assert(argc == 2);
  const int num_threads = std::atoi(argv[1]);

  assert(signal(SIGINT, on_sigint) != SIG_ERR);

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(busy_wait);
  }
  ready = true;
  for (auto& thread : threads) {
    thread.join();
  }
}
