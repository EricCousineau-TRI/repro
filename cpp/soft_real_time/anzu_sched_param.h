// Stuff used to debug some real-time priority stuff on Ubuntu 18.04.
#ifndef _ANZU_SCHED_PARAM_H
  #define _ANZU_SCHED_PARAM_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stdbool.h>
#include <sys/types.h>

#define ANZU_NPROC 64  // HACK: Eric's machine.

typedef struct anzu_sched_param {
  int sched_rr_priority;
  bool cpu_affinity[ANZU_NPROC];
} anzu_sched_param_t;

int anzu_gettid();

// WARNING: No bounds checking!!!
char* anzu_sched_param_sprintf(char* buffer, anzu_sched_param_t* param);

void anzu_set_sched_param(pid_t pid, anzu_sched_param_t* param);

void anzu_get_sched_param_from_pid(pid_t pid, anzu_sched_param_t* param);

#ifdef __cplusplus
} // extern "C"
#endif  // __cplusplus

#endif  // _ANZU_SCHED_PARAM_H
