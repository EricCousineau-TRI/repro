#define _GNU_SOURCE
#include <asm/unistd_64.h>
#include <sched.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anzu_sched_param.h"

static void check(bool value) {
  if (!value) {
    fprintf(stderr, "abort!\n");
    exit(1);
  }
}

int anzu_gettid() {
  return syscall(__NR_gettid);
}

char* anzu_sched_param_sprintf(char* buffer, anzu_sched_param_t* param) {
  check(buffer != NULL);
  check(param != NULL);
  buffer += sprintf(
      buffer,
      "( chrt -r %d  taskset -c ",
      param->sched_rr_priority);
  int num_set = 0;
  for (int i = 0; i < ANZU_NPROC; ++i) {
    num_set += param->cpu_affinity[i];
  }
  if (num_set > 0 && num_set < ANZU_NPROC) {
    for (int i = 0; i < ANZU_NPROC; ++i) {
      if (param->cpu_affinity[i]) {
        buffer += sprintf(buffer, "%d,", i);
      }
    }
  }
  buffer += sprintf(buffer, " )");
  return buffer;
}

void anzu_set_sched_param(pid_t pid, anzu_sched_param_t* param) {
  check(param != NULL);
  if (param->sched_rr_priority != -1) {
    // Ask for the real time scheduler to pump the priority for this thread.
    struct sched_param sch_param;
    sch_param.sched_priority = param->sched_rr_priority;
    check(sched_setscheduler(pid, SCHED_RR, &sch_param) == 0);
  }
  int num_set = 0;
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  for (int i = 0; i < ANZU_NPROC; ++i) {
    if (param->cpu_affinity[i]) {
      num_set += 1;
      CPU_SET(i, &cpu_set);
    }
  }
  if (num_set > 0) {
    check(sched_setaffinity(pid, sizeof(cpu_set), &cpu_set) == 0);
  }
}

void anzu_get_sched_param_from_pid(pid_t pid, anzu_sched_param_t* param) {
  memset(param, 0, sizeof(anzu_sched_param_t));
  // Within narrow usage we have.
  const int sched_policy = sched_getscheduler(pid);
  if (sched_policy == SCHED_RR) {
    struct sched_param sch_param;
    check(sched_getparam(pid, &sch_param) == 0);
    param->sched_rr_priority = sch_param.sched_priority;
  } else {
    check(sched_policy == SCHED_OTHER);
    param->sched_rr_priority = -1;
  }

  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  check(sched_getaffinity(pid, sizeof(cpu_set), &cpu_set) == 0);
  for (int i = 0; i < ANZU_NPROC; ++i) {
    if (CPU_ISSET(i, &cpu_set)) {
      param->cpu_affinity[i] = true;
    }
  }
}
