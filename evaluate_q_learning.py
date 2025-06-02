import time
import random
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

# Register the environment
register(
    id="TaxiTwoPassenger-v0",
    entry_point="multi_taxi:TaxiTwoPassengerEnv",
    max_episode_steps=200,
    reward_threshold=40,
)

# Load environment and q table
env = gym.make("TaxiTwoPassenger-v0", render_mode="human")
Q   = np.load("q_table_two_passenger.npy")

episodes  = 5
max_steps = 200

for ep in range(episodes):
    state, _ = env.reset()
    env.render()
    total_reward = 0
    print(f"\n--- Episode {ep + 1} ---")

    for step in range(max_steps):
        # Greedy action, random tie-break
        q_vals       = Q[state]
        max_q        = q_vals.max()
        best_actions = np.where(np.isclose(q_vals, max_q, atol=1e-8))[0]
        action       = int(random.choice(best_actions))

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        print(f"Step {step + 1}")
        print(f"  State:       {state}")
        print(f"  Action:      {action}")
        print(f"  Reward:      {reward}")
        print(f"  Cum. reward: {total_reward}")
        print(f"  Terminated:  {terminated}, Truncated: {truncated}")
        env.render()
        print("-" * 40)

        state = next_state
        time.sleep(0.4)

        # bail out if the agent implodes
        if step > 50 and total_reward < -100:
            print("⚠️  Agent seems stuck — breaking this episode.")
            break

        if terminated or truncated:
            print(f"✅ Finished in {step + 1} steps — total reward {total_reward}\n")
            break

env.close()
