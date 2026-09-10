#!/usr/bin/env python3
"""
Simple test to verify KeyBoardIntervention functionality
"""
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import time
import sys
from pathlib import Path

# Add the project root to sys.path to allow imports
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ur_env.envs.wrappers import KeyBoardIntervention

class DummyEnv(gym.Env):
    """A minimal dummy environment for testing KeyBoardIntervention"""
    
    def __init__(self):
        super().__init__()
        # Define 7-dim action space (typical for UR robots with gripper)
        self.action_space = Box(
            low=-np.ones(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
        )
        # Simple observation space
        self.observation_space = Box(
            low=-np.ones(10, dtype=np.float32),
            high=np.ones(10, dtype=np.float32),
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.zeros(10, dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs = np.random.randn(10).astype(np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {"action_received": action}
        return obs, reward, terminated, truncated, info

def main():
    print("Creating dummy environment...")
    dummy_env = DummyEnv()
    
    print("Wrapping with KeyBoardIntervention...")
    env = KeyBoardIntervention(dummy_env)
    
    print("\nEnvironment is ready!")
    print("Key bindings:")
    print("- Press ';' to toggle intervention mode ON/OFF")
    print("- When intervention mode is ON:")
    print("  - W/S: X-axis movement")
    print("  - A/D: Y-axis movement") 
    print("  - J/K: Z-axis movement")
    print("  - L: Toggle gripper")
    print("\nPress Ctrl+C to exit")
    
    # Just run indefinitely to allow keyboard interaction
    try:
        obs, info = env.reset()
        step_count = 0
        
        while True:
            # Use zero action as base
            action = np.zeros(7, dtype=np.float32)
            
            # Step through environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Show if there was an intervention
            if "intervene_action" in info:
                intervene_action = info["intervene_action"]
                print(f"Intervention! Action: [{intervene_action[0]:.2f}, {intervene_action[1]:.2f}, {intervene_action[2]:.2f}], Gripper: {intervene_action[6]:.2f}")
            
            step_count += 1
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nExiting...")
        return

if __name__ == "__main__":
    main()