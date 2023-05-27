# Threading / Multiprocessing with Drake Systems

## Setup

Ensure you have `poetry` installed.

Install prereqs

```
poetry install
```

When using scripts, be sure to call `poetry shell`.

## Script Setup

- This creates a diagram with N "busy" systems that are delegated to workers,
  and (optionally) a printing system in the direct diagram/process.
- Thus "busy" system has a busy-loop to try and burn time for its given intended
  period (according to wall-clock time)
- The systems are stepped by the worker using mailboxing (loosely based on Anzu
  code).

## Prelim Analysis

Based on results below

- Outputs match for single-system case (and should match for multi-system case), and
  should approximate a zero-order hold.
- Care is taken to only initialize at initialization events, and *not* perform any
  extra updates during initialization.
- Rate for single-system + 10 Hz case is ~1x across the board
- Rate for 5x system + 500 Hz case is 1/5x for direct, 1/5x for threaded, and
  ~4/5x for multiprocess worker.
- Threading may get bottlenecked by busy Python work locking up the GIL
- I (Eric) am not sure why multiprocess setup seems to run faster than
  realtime...

## Running

Example

```sh
# Example output (2023-05-26):
$ python -m thread_system.main

DirectWorker, num_systems=1, deterministic=True
  period_sec=0.1, t_sim=0.4
[0 print] t=0, y=1
[0 print] t=0.1, y=1
[0 print] t=0.2, y=1.1
[0 print] t=0.3, y=1.2
Rate: 0.998
y: 1.2

ThreadWorker, num_systems=1, deterministic=True
  period_sec=0.1, t_sim=0.4
[0 print] t=0, y=1
[0 print] t=0.1, y=1
[0 print] t=0.2, y=1.1
[0 print] t=0.3, y=1.2
Rate: 0.985
y: 1.2

MultiprocessWorker, num_systems=1, deterministic=True
  period_sec=0.1, t_sim=0.4
[0 print] t=0, y=1
[0 print] t=0.1, y=1
[0 print] t=0.2, y=1.1
[0 print] t=0.3, y=1.2
Rate: 1.32
y: 1.2

DirectWorker, num_systems=5, deterministic=True
  period_sec=0.002, t_sim=1
Rate: 0.193
y: 2

ThreadWorker, num_systems=5, deterministic=True
  period_sec=0.002, t_sim=1
Rate: 0.163
y: 2

MultiprocessWorker, num_systems=5, deterministic=True
  period_sec=0.002, t_sim=1
Rate: 0.825
y: 2
```
