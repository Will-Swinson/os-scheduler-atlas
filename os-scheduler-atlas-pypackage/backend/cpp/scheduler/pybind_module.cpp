#include "scheduler.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace scheduler;

Process dict_to_process(py::dict proc_dict) {
  return Process{proc_dict["pid"].cast<int>(),
                 proc_dict["arrival_time"].cast<int>(),
                 proc_dict["burst_time"].cast<int>()};
}

py::dict process_to_dict(const Process &process) {
  py::dict result;

  result["pid"] = process.pid;
  result["arrival_time"] = process.arrivalTime;
  result["burst_time"] = process.burstTime;
  result["waiting_time"] = process.waitingTime;
  result["turn_around_time"] = process.turnaroundTime;
  result["finish_time"] = process.finishTime;
  result["remaining_time"] = process.remainingTime;
  result["is_complete"] = process.isComplete;

  return result;
}

/**
 * @brief Schedule a list of processes using First Come First Served and return the scheduled processes.
 *
 * Converts a Python list of process dictionaries into native Process objects, runs the FCFS scheduler,
 * and converts the scheduled Process objects back into a Python list of dictionaries.
 *
 * @param process_list Python list of dictionaries, each representing a process and containing at minimum the keys
 *                     `pid`, `arrival_time`, and `burst_time`.
 * @return py::list Python list of dictionaries representing the scheduled processes. Each dictionary includes
 *                 `pid`, `arrival_time`, `burst_time`, and scheduling result fields such as `waiting_time`,
 *                 `turn_around_time`, `finish_time`, `remaining_time`, and `is_complete`.
 */
py::list fcfs_scheduler_wrapper(py::list process_list) {
  std::vector<Process> processes;

  for (const auto &process : process_list) {
    Process convertedProcess = dict_to_process(process.cast<py::dict>());
    processes.push_back(convertedProcess);
  }

  std::vector<Process> result = fcfsScheduler(processes);

  py::list result_process_list;

  for (const auto &process : result) {
    py::dict convertedProcess = process_to_dict(process);
    result_process_list.append(convertedProcess);
  }

  return result_process_list;
}

/**
 * @brief Runs Shortest Job First scheduling on a list of process dictionaries.
 *
 * @param process_list Python list of dicts where each dict represents a process and must contain integer keys `pid`, `arrival_time`, and `burst_time`.
 * @return py::list Python list of dicts representing the scheduled processes; each dict includes `pid`, `arrival_time`, `burst_time`, `waiting_time`, `turn_around_time`, `finish_time`, `remaining_time`, and `is_complete`.
 */
py::list sjf_scheduler_wrapper(py::list process_list) {
  std::vector<Process> processes;

  for (const auto &process : process_list) {
    Process convertedProcess = dict_to_process(process.cast<py::dict>());
    processes.push_back(convertedProcess);
  }

  std::vector<Process> result = sjfScheduler(processes);

  py::list result_process_list;

  for (const auto &process : result) {
    py::dict convertedProcess = process_to_dict(process);
    result_process_list.append(convertedProcess);
  }

  return result_process_list;
}

/**
 * @brief Runs the Round Robin scheduling algorithm on a list of processes.
 *
 * Converts a Python list of process dictionaries into Process objects, executes
 * the Round Robin scheduler with the given time quantum, and returns the
 * scheduled processes as a Python list of dictionaries.
 *
 * @param process_list Python list of dicts where each dict contains process fields (e.g., `pid`, `arrival_time`, `burst_time`).
 * @param time_quantum Time slice, in the same time units used by the process fields, to use for the Round Robin scheduler.
 * @return py::list Python list of dicts representing processes after scheduling with updated timing and completion fields.
 */
py::list round_robin_scheduler_wrapper(py::list process_list,
                                       int time_quantum) {
  std::vector<Process> processes;

  for (const auto &process : process_list) {
    Process convertedProcess = dict_to_process(process.cast<py::dict>());
    processes.push_back(convertedProcess);
  }

  std::vector<Process> result = roundRobinScheduler(processes, time_quantum);

  py::list result_process_list;

  for (const auto &process : result) {
    py::dict convertedProcess = process_to_dict(process);
    result_process_list.append(convertedProcess);
  }

  return result_process_list;
}

/**
 * @brief Defines the Python module "scheduler_cpp" and registers scheduling functions.
 *
 * Initializes the pybind11 module named "scheduler_cpp" with documentation and exposes
 * three scheduler bindings: `fcfs_scheduler`, `sjf_scheduler`, and `round_robin_scheduler`.
 *
 * - `fcfs_scheduler(processes)`: First Come First Served scheduling algorithm.
 * - `sjf_scheduler(processes)`: Shortest Job First scheduling algorithm.
 * - `round_robin_scheduler(processes, time_quantum)`: Round Robin scheduling algorithm with a time quantum.
 */
PYBIND11_MODULE(scheduler_cpp, m) {
  m.doc() = "OS Scheduling Algorithms";

  m.def("fcfs_scheduler", &fcfs_scheduler_wrapper,
        "First Come First Served scheduling algorithm", py::arg("processes"));

  m.def("sjf_scheduler", &sjf_scheduler_wrapper,
        "Shortest Job First scheduling algorithm", py::arg("processes"));

  m.def("round_robin_scheduler", &round_robin_scheduler_wrapper,
        "Round Robin scheduling algorithm", py::arg("processes"),
        py::arg("time_quantum"));
}