import argparse
import time

import numpy as np
import requests


def main():
    parser = argparse.ArgumentParser(description="Inspect idle TCP force/torque from the UR server.")
    parser.add_argument("--host", default="127.0.0.1", help="UR server host")
    parser.add_argument("--port", type=int, default=5000, help="UR server port")
    parser.add_argument("--hz", type=float, default=10.0, help="Polling frequency")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to collect")
    parser.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/getstate"
    dt = 1.0 / args.hz
    wrench_history = []

    print(f"Polling {url} at {args.hz:.2f} Hz for {args.samples} samples")
    print("idx | force[N] | torque[Nm] | wrench[Fx Fy Fz Tx Ty Tz]")

    for idx in range(args.samples):
        start = time.monotonic()
        resp = requests.post(url, timeout=args.timeout)
        resp.raise_for_status()
        state = resp.json()

        force = np.asarray(state.get("force", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        torque = np.asarray(state.get("torque", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        wrench = np.concatenate([force, torque], axis=0)
        wrench_history.append(wrench)

        force_str = np.array2string(force, precision=4, separator=", ")
        torque_str = np.array2string(torque, precision=4, separator=", ")
        wrench_str = np.array2string(wrench, precision=4, separator=", ")
        print(f"{idx:04d} | {force_str} | {torque_str} | {wrench_str}")

        elapsed = time.monotonic() - start
        if idx + 1 < args.samples and dt > elapsed:
            time.sleep(dt - elapsed)

    wrench_arr = np.asarray(wrench_history, dtype=np.float64)
    mean = wrench_arr.mean(axis=0)
    std = wrench_arr.std(axis=0)
    min_v = wrench_arr.min(axis=0)
    max_v = wrench_arr.max(axis=0)

    print("\nSummary")
    print(f"mean wrench : {np.array2string(mean, precision=5, separator=', ')}")
    print(f"std wrench  : {np.array2string(std, precision=5, separator=', ')}")
    print(f"min wrench  : {np.array2string(min_v, precision=5, separator=', ')}")
    print(f"max wrench  : {np.array2string(max_v, precision=5, separator=', ')}")


if __name__ == "__main__":
    main()
