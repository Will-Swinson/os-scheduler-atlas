#pragma once
#include <string>
#include <vector>

namespace scheduler {

struct Process {
  int pid;
  int arrivalTime;
  int burstTime;

  int startTime;
  int finishTime;
  int waitingTime;
  int turnaroundTime;

  Process(int id, int arrival, int burst)
      : pid(id), arrivalTime(arrival), burstTime(burst), startTime(0),
        finishTime(0), waitingTime(0), turnaroundTime(0) {};
};

std::vector<Process> fcfsScheduler(const std::vector<Process> &processes);
} // namespace scheduler