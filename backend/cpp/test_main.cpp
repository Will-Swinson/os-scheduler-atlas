#include "scheduler.h"
#include <algorithm>
#include <iostream>

int main() {

  std::vector<scheduler::Process> processes;

  // Our test data: A(0,5), B(2,3), C(4,2)
  processes.push_back(scheduler::Process(1, 0, 5));
  processes.push_back(scheduler::Process(2, 2, 3));
  processes.push_back(scheduler::Process(3, 4, 2));

  // Run FCFS
  std::vector<scheduler::Process> result = scheduler::fcfsScheduler(processes);

  // Print results (need #include <iostream>)
  for (const auto &p : result) {
    std::cout << "Process " << p.pid << ": start=" << p.startTime
              << ", finish=" << p.finishTime << ", wait=" << p.waitingTime
              << std::endl;
  }

  return 0;
};