#!/usr/bin/env python3
"""Test whether JAX is using CUDA/GPU or CPU."""

import jax

def main():
    print(f"JAX version: {jax.__version__}")
    print(f"JAX platform: {jax.default_backend()}")  # 'cpu', 'gpu', 'tpu'

    devices = jax.devices()
    print(f"Number of devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device.device_kind} ({device.platform})")

    # Try a simple computation
    try:
        a = jax.numpy.array([1.0, 2.0, 3.0])
        b = a + 1.0
        print(f"Computation successful: {b}")
        print(f"Array device: {b.device()}")
    except Exception as e:
        print(f"Computation failed: {e}")

if __name__ == "__main__":
    main()