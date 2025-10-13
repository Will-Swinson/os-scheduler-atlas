#include "scheduler.h"
#include <gtest/gtest.h>

using scheduler::fcfsScheduler;
using scheduler::Process;
using scheduler::roundRobinScheduler;
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

TEST(RoundRobinScheduler, CalculateCorrectFinishTimes) {
  std::vector<Process> processes;

  // Arguments are pid, arrival, and burst time.
  processes.push_back(Process{1, 0, 8}); // long process
  processes.push_back(Process{2, 1, 1}); // Short process

  std::vector<Process> result = roundRobinScheduler(processes, 3);

  EXPECT_EQ(result[0].finishTime, 9);
  EXPECT_EQ(result[1].finishTime, 4);
}

TEST(RoundRobinScheduler, CalculateCorrectWaitingTimes) {
  std::vector<Process> processes;

  // Arguments are pid, arrival, and burst time.
  processes.push_back(Process{1, 0, 8}); // long process
  processes.push_back(Process{2, 1, 1}); // Short process

  std::vector<Process> result = roundRobinScheduler(processes, 3);

  EXPECT_EQ(result[0].waitingTime, 1);
  EXPECT_EQ(result[1].waitingTime, 2);
}

TEST(RoundRobinScheduler, HandlesEmptyProcessList) {
  std::vector<Process> emptyProcessList{};

  std::vector<Process> result = roundRobinScheduler(emptyProcessList, 3);

  EXPECT_TRUE(result.empty());
}