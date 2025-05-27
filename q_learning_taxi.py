import gymnasium as gym
import numpy as np
import random
from multi_taxi import TaxiTwoPassengerEnv

# Hyperparameters
alpha = 0.1          # learning rate
gamma = 0.99         # discount factor
epsilon = 1.0        # exploration rate
epsilon_decay = 0.995
min_epsilon = 0.05
episodes = 10000
max_steps = 200

# Initialize environment
env = gym.make("TaxiTwoPassenger-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions))

# Training loop
for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state
        total_reward += reward

        if done:
            break

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Print progress
    if (ep + 1) % 1000 == 0:
        print(f"Episode {ep + 1}: Total reward = {total_reward}")

# Save Q-table
np.save("q_table_two_passenger.npy", Q)
print("Training finished.")
