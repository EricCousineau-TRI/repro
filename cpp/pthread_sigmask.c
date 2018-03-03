#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <cassert>

#include <iostream>
#include <thread>

/* Simple error handling functions */

class SigintThread {
 public:
  SigintThread() {
    // Block SIGINT using a main thread.
    sigemptyset(&set_);
    sigaddset(&set_, SIGINT);
    assert(pthread_sigmask(SIG_BLOCK, &set_, NULL) == 0);
    assert(
        pthread_create(&sig_thread_, NULL, &SigintThread::loop, &set_) == 0);
    std::cerr << "Installed sig_thread\n";
  }

  ~SigintThread() {
    std::cerr << "Destroying sig_thread\n";
    done_ = true;
    void* result = nullptr;
    pthread_join(sig_thread_, &result);
    std::cerr << "Done\n";
  }

  static void* loop(void* arg) {
    sigset_t* set = static_cast<sigset_t*>(arg);
    timespec t{};
    t.tv_sec = 0;
    t.tv_nsec = 100e3;  // 10ms
    while (!done_) {
      std::cerr << "Waiting\n";
      // if (sigtimedwait(set, nullptr, &t) == 0) {
      int sig{};
      sigwait(set, &sig);
      {
        constexpr int return_code = 130;
        std::cerr << "exit(" << return_code << ") on SIGINT" << std::endl;
        exit(return_code);
      }
    }
    return nullptr;
  }

 private:
  static bool done_;
  sigset_t set_;
  pthread_t sig_thread_;
};

bool SigintThread::done_{};

void* other_thread(void* x) {
  std::cout << "pausing" << std::endl;
   pause();            /* Dummy pause so we can test program */
  return NULL;
}

int
main(int argc, char *argv[])
{
   pthread_t thread;
   sigset_t set;
   int s;

   /* Block SIGQUIT and SIGUSR1; other threads created by main()
      will inherit a copy of the signal mask. */

   sigemptyset(&set);
   // sigaddset(&set, SIGQUIT);
   // sigaddset(&set, SIGUSR1);
   sigaddset(&set, SIGINT);
   s = pthread_sigmask(SIG_BLOCK, &set, NULL);
   assert(s == 0);

   SigintThread sig_thread;

  std::thread thread_2(&other_thread, nullptr);
   /* Main thread carries on to create other threads and/or do
      other work */
   thread_2.join();
}
