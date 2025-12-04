def calculate_avg_metrics(scheduler_output):
    if not scheduler_output:
        raise ValueError("scheduler_output must be non-empty")

    avg_waiting_time = sum(p["waiting_time"] for p in scheduler_output) / len(
        scheduler_output
    )
    avg_turnaround_time = sum(p["turn_around_time"] for p in scheduler_output) / len(
        scheduler_output
    )

    return avg_waiting_time, avg_turnaround_time
