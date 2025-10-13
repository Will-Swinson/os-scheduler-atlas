#include "scheduler.h"
#include <gtest/gtest.h>

using scheduler::fcfsScheduler;
using scheduler::Process;
using scheduler::roundRobinScheduler;
using scheduler::sjfScheduler;

TEST(SystemTest, ComplexWorkloadAllAlgorithms) {
  std::vector<Process> workload;

  workload.push_back(Process{1, 0, 2});
  workload.push_back(Process{2, 1, 1});

  workload.push_back(Process{3, 2, 4});
  workload.push_back(Process{4, 3, 3});

  workload.push_back(Process{5, 4, 12});
  workload.push_back(Process{6, 5, 8});

  auto fcfsResult = fcfsScheduler(workload);
  auto sjfResult = sjfScheduler(workload);
  auto rrResult = roundRobinScheduler(workload, 3);

  EXPECT_EQ(fcfsResult.size(), 6);
  EXPECT_EQ(sjfResult.size(), 6);
  EXPECT_EQ(rrResult.size(), 6);

  for (const auto &process : fcfsResult) {
    EXPECT_TRUE(process.isComplete);
    EXPECT_EQ(process.remainingTime, 0);
  }

  for (const auto &process : sjfResult) {
    EXPECT_TRUE(process.isComplete);
    EXPECT_EQ(process.remainingTime, 0);
  }

  for (const auto &process : rrResult) {
    EXPECT_TRUE(process.isComplete);
    EXPECT_EQ(process.remainingTime, 0);
  }

  int maxWaitFCFS = 0;
  int maxWaitSJF = 0;
  int maxWaitRR = 0;

  for (int i = 0; i < 6; i++) {
    maxWaitFCFS = std::max(maxWaitFCFS, fcfsResult[i].waitingTime);
    maxWaitSJF = std::max(maxWaitSJF, sjfResult[i].waitingTime);
    maxWaitRR = std::max(maxWaitRR, rrResult[i].waitingTime);
  }

  EXPECT_LT(maxWaitRR, maxWaitFCFS);
  EXPECT_LT(maxWaitRR, maxWaitSJF);
}