#include "scheduler.h"
#include <gtest/gtest.h>

using scheduler::fcfsScheduler;
using scheduler::Process;
using scheduler::roundRobinScheduler;
using scheduler::sjfScheduler;

TEST(SchedulerComparison, FairnessComparisonFCFSvsRoundRobin) {
  std::vector<Process> processes;

  processes.push_back(Process{1, 0, 10});
  processes.push_back(Process{2, 1, 2});
  processes.push_back(Process{3, 2, 1});

  auto fcfsResult = fcfsScheduler(processes);
  auto roundRobinResult = roundRobinScheduler(processes, 3);

  EXPECT_LT(roundRobinResult[1].waitingTime, fcfsResult[1].waitingTime);
  EXPECT_LT(roundRobinResult[2].waitingTime, fcfsResult[2].waitingTime);
}