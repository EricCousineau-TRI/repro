#include <pthread.h>
#include <stdio.h>
#include <string.h>

// #include <glib.h>

#include "anzu_sched_param.h"

static int get_custom_pid() {
  return 0;
  // return anzu_gettid();
}

static void check(bool value) {
  if (!value) {
    fprintf(stderr, "abort!\n");
    exit(1);
  }
}

typedef void* (*thread_func_t)(void*);

#define USE_PTHREAD

static void run_as_thread(thread_func_t func) {
  void* user_data = NULL;
#ifdef USE_PTHREAD  // pthread
  pthread_t thread_id;
  pthread_attr_t thread_attr;
  check(pthread_attr_init(&thread_attr) == 0);
  check(pthread_create(&thread_id, &thread_attr, func, user_data) == 0);
  void* thread_retval;
  check(pthread_join(thread_id, &thread_retval) == 0);
  check(thread_retval == NULL);
#else  // gthread, deprecated
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  GThread* gthread = g_thread_create(func, user_data, true, NULL);
#pragma GCC diagnostic pop
  const void* gthread_retval = g_thread_join(gthread);
  check(gthread_retval == NULL);
#endif  // USE_PTHREAD
}

static void* run_get_thread(void* raw) {
  char buffer[1024];
  check(raw == NULL);
  const pid_t pid = get_custom_pid();
  printf("run_get_thread: tid=%d\n", anzu_gettid());

  anzu_sched_param_t init_param;
  anzu_get_sched_param_from_pid(pid, &init_param);
  anzu_sched_param_sprintf(buffer, &init_param);
  printf("  get(%d) --> %s\n", pid, buffer);

  return NULL;
}

static void* run_inner_get_thread(void* raw) {
  char buffer[1024];
  check(raw == NULL);
  printf("  run_inner_get_thread: tid=%d\n", anzu_gettid());
  const pid_t pid = get_custom_pid();

  anzu_sched_param_t init_param;
  anzu_get_sched_param_from_pid(pid, &init_param);
  anzu_sched_param_sprintf(buffer, &init_param);
  printf("    get(%d) --> %s\n", pid, buffer);

  return NULL;
}

static void* run_get_set_get_thread(void* raw) {
  char buffer[1024];
  check(raw == NULL);
  const pid_t pid = get_custom_pid();
  printf("run_get_set_get_thread: tid=%d\n", anzu_gettid());

  anzu_sched_param_t init_param;
  anzu_get_sched_param_from_pid(pid, &init_param);
  anzu_sched_param_sprintf(buffer, &init_param);
  printf("  get(%d) --> %s\n", pid, buffer);

  anzu_sched_param_t user_param;
  memset(&user_param, 0, sizeof(anzu_sched_param_t));
  user_param.sched_rr_priority = 30;
  user_param.cpu_affinity[3] = true;
  anzu_sched_param_sprintf(buffer, &user_param);
  printf("  SET(%d): %s\n", pid, buffer);
  anzu_set_sched_param(pid, &user_param);

  anzu_sched_param_t post_param;
  anzu_get_sched_param_from_pid(pid, &post_param);
  anzu_sched_param_sprintf(buffer, &post_param);
  printf("  get(%d) --> %s\n", pid, buffer);

  run_as_thread(run_inner_get_thread);

  return NULL;
}

int main(int argc, char** argv) {
  char buffer[1024];
  const pid_t pid = get_custom_pid();

  anzu_sched_param_t init_param;
  anzu_get_sched_param_from_pid(pid, &init_param);
  anzu_sched_param_sprintf(buffer, &init_param);
  printf("get(%d) --> %s\n", pid, buffer);

  run_as_thread(run_get_set_get_thread);
  run_as_thread(run_get_thread);

  anzu_sched_param_t post_param;
  anzu_get_sched_param_from_pid(pid, &post_param);
  anzu_sched_param_sprintf(buffer, &post_param);
  printf("get(%d) --> %s\n", pid, buffer);

  printf("\n");

  anzu_sched_param_t user_param2;
  memset(&user_param2, 0, sizeof(anzu_sched_param_t));
  user_param2.sched_rr_priority = 20;
  user_param2.cpu_affinity[2] = true;
  anzu_sched_param_sprintf(buffer, &user_param2);
  printf("SET(%d): %s\n", pid, buffer);
  anzu_set_sched_param(pid, &user_param2);

  run_as_thread(run_get_set_get_thread);
  run_as_thread(run_get_thread);

  anzu_sched_param_t post_param2;
  anzu_get_sched_param_from_pid(pid, &post_param2);
  anzu_sched_param_sprintf(buffer, &post_param2);
  printf("get(%d) --> %s\n", pid, buffer);

  printf("\n");
  return 0;
}
