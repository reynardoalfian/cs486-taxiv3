import numpy as np
import gymnasium as gym
import time
from multi_taxi import TaxiTwoPassengerEnv

# Load environment and Q-table
env = gym.make("TaxiTwoPassenger-v0", render_mode="human")
Q = np.load("q_table_two_passenger.npy")

episodes = 5
max_steps = 200

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    print(f"\n--- Episode {ep + 1} ---")

    for step in range(max_steps):
        env.render()
        time.sleep(0.5)

        action = np.argmax(Q[state])
        print(f"Step {step + 1}:")
        print(f"  State: {state}")
        print(f"  Best Action: {action}")
        print(f"  Q-values: {Q[state]}")

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state

        print(f"  Reward: {reward}")
        print(f"  Total reward so far: {total_reward}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}\n")

        # Emergency break condition if stuck
        if step > 50 and total_reward < -100:
            print("⚠️  Agent likely stuck in a loop or bad policy. Breaking episode.")
            break

        if terminated or truncated:
            print(f"✅ Episode finished in {step + 1} steps with total reward: {total_reward}")
            break

env.close()
