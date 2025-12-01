def calculate_avg_metrics(scheduler_output):
    avg_waiting_time = sum(p["waiting_time"] for p in scheduler_output) / len(
        scheduler_output
    )
    avg_turnaround_time = sum(p["turn_around_time"] for p in scheduler_output) / len(
        scheduler_output
    )

    return avg_waiting_time, avg_turnaround_time
