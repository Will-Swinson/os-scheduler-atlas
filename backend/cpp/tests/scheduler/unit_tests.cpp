#include "scheduler.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <iostream>

using scheduler::fcfsScheduler;
using scheduler::Process;
using scheduler::sjfScheduler;

TEST(FCFSScheduler, CalculatesCorrectFinishTimes) {
  std::vector<Process> processes;

  // Arguments are pid, arrival, and burst time.
  processes.push_back(Process{1, 0, 5});
  processes.push_back(Process{2, 2, 3});
  processes.push_back(Process{3, 4, 2});

  std::vector<Process> result = fcfsScheduler(processes);

  EXPECT_EQ(result[0].finishTime, 5);
  EXPECT_EQ(result[1].finishTime, 8);
  EXPECT_EQ(result[2].finishTime, 10);
}

TEST(FCFSScheduler, HandlesEmptyProcessList) {
  std::vector<Process> emptyProcessList{};

  std::vector<Process> result = fcfsScheduler(emptyProcessList);

  EXPECT_TRUE(result.empty());
}

// SJF Scheduler Unit Tests
TEST(SJFScheduler, CalculatesCorrectFinishTimes) {
  std::vector<Process> processes;

  // Arguments are pid, arrival, and burst time.
  processes.push_back(Process{1, 0, 5});
  processes.push_back(Process{2, 2, 3});
  processes.push_back(Process{3, 4, 2});

  std::vector<Process> result = sjfScheduler(processes);

  EXPECT_EQ(result[0].finishTime, 6);
  EXPECT_EQ(result[1].finishTime, 9);
  EXPECT_EQ(result[2].finishTime, 14);
}

TEST(SJFScheduler, HandlesEmptyProcessList) {
  std::vector<Process> emptyProcessList{};

  std::vector<Process> result = sjfScheduler(emptyProcessList);

  EXPECT_TRUE(result.empty());
}