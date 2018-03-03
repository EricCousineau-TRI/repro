#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>

/* Simple error handling functions */

#define handle_error_en(en, msg) \
       do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

static void *
sig_thread(void *arg)
{
   sigset_t *set = arg;
   int s, sig;

   for (;;) {
       s = sigwait(set, &sig);
       if (s != 0)
           handle_error_en(s, "sigwait");
       printf("Signal handling thread got signal %d\n", sig);
   }
}

void* other_thread(void* x) {
   pause();            /* Dummy pause so we can test program */
  return NULL;
}

int
main(int argc, char *argv[])
{
   pthread_t thread;
   pthread_t thread_2;
   sigset_t set;
   int s;

   /* Block SIGQUIT and SIGUSR1; other threads created by main()
      will inherit a copy of the signal mask. */

   sigemptyset(&set);
   // sigaddset(&set, SIGQUIT);
   // sigaddset(&set, SIGUSR1);
   sigaddset(&set, SIGINT);
   s = pthread_sigmask(SIG_BLOCK, &set, NULL);
   if (s != 0)
       handle_error_en(s, "pthread_sigmask");

   s = pthread_create(&thread, NULL, &sig_thread, (void *) &set);
   if (s != 0)
       handle_error_en(s, "pthread_create");

   s = pthread_create(&thread_2, NULL, &other_thread, NULL);
   if (s != 0)
       handle_error_en(s, "pthread_create");

   /* Main thread carries on to create other threads and/or do
      other work */
   pthread_join(thread_2, NULL);
}
