#include "scheduler.h"
#include <algorithm>
#include <iostream>
namespace scheduler {
std::vector<Process> fcfsScheduler(const std::vector<Process> &processes) {
  std::vector<Process> result = processes;

  std::sort(result.begin(), result.end(),
            [](const Process &a, const Process &b) {
              return a.arrivalTime < b.arrivalTime;
            });

  int currentTime = 0;

  for (int i = 0; i < result.size(); i++) {
    Process &currProcess = result[i];

    currProcess.startTime = std::max(currentTime, currProcess.arrivalTime);

    currProcess.finishTime = currProcess.startTime + currProcess.burstTime;

    currProcess.waitingTime = currProcess.startTime - currProcess.arrivalTime;
    currProcess.turnaroundTime =
        currProcess.finishTime - currProcess.arrivalTime;

    currentTime = currProcess.finishTime;
  };

  return result;
};
} // namespace scheduler

