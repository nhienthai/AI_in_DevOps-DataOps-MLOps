"""Load testing script for the Movie Rating API."""

import argparse
import concurrent.futures
import random
import statistics
import time
from typing import Callable, List, Tuple

import requests


API_URL = "http://localhost:8000"


def make_single_prediction() -> Tuple[bool, float]:
    """Make a single prediction request."""
    user_id = str(random.randint(1, 943))
    movie_id = str(random.randint(1, 1682))
    start_time = time.perf_counter()

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"user_id": user_id, "movie_id": movie_id},
            timeout=5,
        )
        return response.status_code == 200, (time.perf_counter() - start_time) * 1000
    except Exception:
        return False, (time.perf_counter() - start_time) * 1000


def make_batch_prediction(batch_size: int = 10) -> Tuple[bool, float]:
    """Make a batch prediction request."""
    predictions = [
        {
            "user_id": str(random.randint(1, 943)),
            "movie_id": str(random.randint(1, 1682)),
        }
        for _ in range(batch_size)
    ]
    start_time = time.perf_counter()

    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"predictions": predictions},
            timeout=10,
        )
        return response.status_code == 200, (time.perf_counter() - start_time) * 1000
    except Exception:
        return False, (time.perf_counter() - start_time) * 1000


def check_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def _run_worker_cycle(
    executor: concurrent.futures.ThreadPoolExecutor,
    workers: int,
    request_fn: Callable[[], Tuple[bool, float]],
) -> Tuple[int, int, List[float]]:
    futures = [executor.submit(request_fn) for _ in range(workers)]
    total = 0
    successful = 0
    latencies: List[float] = []

    for future in concurrent.futures.as_completed(futures):
        success, latency = future.result()
        total += 1
        successful += int(success)
        latencies.append(latency)

    return total, successful, latencies


def run_load_test(
    duration: int = 60,
    workers: int = 10,
    batch_mode: bool = False,
    allow_unhealthy: bool = False,
):
    """Run a steady load test for the specified duration."""
    print("=" * 60)
    print("Load Test for Movie Rating API")
    print("=" * 60)
    print(f"Duration: {duration}s")
    print(f"Workers: {workers}")
    print(f"Mode: {'Batch' if batch_mode else 'Single'}")
    print("=" * 60)

    if not allow_unhealthy and not check_health():
        print("ERROR: API is not healthy. Aborting load test.")
        return

    if allow_unhealthy:
        print("WARNING: Skipping health check so failed requests can be generated.")

    request_fn: Callable[[], Tuple[bool, float]] = (
        lambda: make_batch_prediction(10) if batch_mode else make_single_prediction()
    )

    total_requests = 0
    successful = 0
    latencies: List[float] = []
    start_time = time.time()
    next_report = 10

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while time.time() - start_time < duration:
            total, ok, cycle_latencies = _run_worker_cycle(executor, workers, request_fn)
            total_requests += total
            successful += ok
            latencies.extend(cycle_latencies)

            elapsed = int(time.time() - start_time)
            if elapsed >= next_report:
                print(f"  Progress: {elapsed}s - {total_requests} requests")
                next_report += 10

            time.sleep(0.1)

    print_statistics(total_requests, successful, latencies, duration)


def print_statistics(total: int, successful: int, latencies: list, duration: int):
    """Print load test statistics."""
    print("\n" + "=" * 60)
    print("Load Test Results")
    print("=" * 60)
    print(f"Total Requests:    {total}")
    print(f"Successful:        {successful}")
    print(f"Failed:            {total - successful}")
    print(f"Success Rate:      {successful / total * 100:.2f}%" if total > 0 else "N/A")
    print(f"Requests/Second:   {total / duration:.2f}")

    if latencies:
        print("\nLatency Statistics (ms):")
        print(f"  Min:    {min(latencies):.2f}")
        print(f"  Max:    {max(latencies):.2f}")
        print(f"  Mean:   {statistics.mean(latencies):.2f}")
        print(f"  Median: {statistics.median(latencies):.2f}")
        if len(latencies) > 1:
            print(f"  StdDev: {statistics.stdev(latencies):.2f}")

        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        print(f"  P50:    {p50:.2f}")
        print(f"  P95:    {p95:.2f}")
        print(f"  P99:    {p99:.2f}")

    print("=" * 60)


def run_variable_load(duration: int = 120, max_workers: int = 50, allow_unhealthy: bool = False):
    """Run a variable load pattern with ramp up, steady state, and ramp down."""
    print("Running variable load pattern")
    if not allow_unhealthy and not check_health():
        print("ERROR: API is not healthy. Aborting load test.")
        return

    if allow_unhealthy:
        print("WARNING: Skipping health check so failed requests can be generated.")

    ramp_duration = 30
    steady_duration = max(0, duration - 2 * ramp_duration)

    for second in range(ramp_duration):
        workers = max(1, int(round(1 + (max_workers - 1) * (second + 1) / ramp_duration)))
        run_load_test(duration=1, workers=workers, allow_unhealthy=allow_unhealthy)

    if steady_duration > 0:
        run_load_test(duration=steady_duration, workers=max_workers, allow_unhealthy=allow_unhealthy)

    for second in range(ramp_duration):
        workers = max(1, int(round(max_workers - (max_workers - 1) * (second + 1) / ramp_duration)))
        run_load_test(duration=1, workers=workers, allow_unhealthy=allow_unhealthy)


def run_spike_test(
    normal_workers: int = 5,
    spike_workers: int = 100,
    spike_duration: int = 10,
    allow_unhealthy: bool = False,
):
    """Run a spike test with a sudden burst of traffic."""
    print("Running spike test")
    if not allow_unhealthy and not check_health():
        print("ERROR: API is not healthy. Aborting load test.")
        return

    if allow_unhealthy:
        print("WARNING: Skipping health check so failed requests can be generated.")

    print("\nPhase 1: normal load")
    run_load_test(duration=30, workers=normal_workers, allow_unhealthy=allow_unhealthy)

    print("\nPhase 2: spike load")
    run_load_test(duration=spike_duration, workers=spike_workers, allow_unhealthy=allow_unhealthy)

    print("\nPhase 3: recovery load")
    run_load_test(duration=30, workers=normal_workers, allow_unhealthy=allow_unhealthy)


def main():
    parser = argparse.ArgumentParser(description="Load test the Movie Rating API")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--batch", action="store_true", help="Use batch predictions")
    parser.add_argument("--variable", action="store_true", help="Run variable load pattern")
    parser.add_argument("--spike", action="store_true", help="Run spike test")
    parser.add_argument(
        "--allow-unhealthy",
        action="store_true",
        help="Keep sending requests even if /health reports unhealthy",
    )

    args = parser.parse_args()

    if args.variable:
        run_variable_load(args.duration, args.workers, args.allow_unhealthy)
    elif args.spike:
        run_spike_test(normal_workers=args.workers, allow_unhealthy=args.allow_unhealthy)
    else:
        run_load_test(args.duration, args.workers, args.batch, args.allow_unhealthy)


if __name__ == "__main__":
    main()
