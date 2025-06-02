import gymnasium as gym
import numpy as np
import random
from multi_taxi import TaxiTwoPassengerEnv

# Hyperparameters
alpha         = 0.1       # learning rate
gamma         = 0.99      # discount factor
epsilon       = 1.0       # initial exploration rate (epsilon greedy)
epsilon_decay = 0.9998    # decay epsilon more slowly to allow more exploration for longer
min_epsilon   = 0.01      # floor for epsilon
episodes      = 100000    # increased total training episodes for better convergence
max_steps     = 200       # max steps per episode (env caps at 200 anyway)

# Set up environment and Q‐table
env = gym.make("TaxiTwoPassenger-v0")
n_states  = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions), dtype=np.float32)

# Training loop
for ep in range(episodes):
    state, _     = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            q_vals = Q[state]
            max_q  = np.max(q_vals)

            candidates = np.where(np.isclose(q_vals, max_q, atol=1e-8))[0]
            action = int(random.choice(candidates))

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update
        max_q_next = np.max(Q[next_state])
        Q[state, action] += alpha * (reward + gamma * max_q_next - Q[state, action])

        state = next_state
        total_reward += reward
        if done:
            break

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Print progress
    if (ep + 1) % 3000 == 0:
        print(f"Episode {ep + 1:>5}/{episodes}: epsilon={epsilon:.3f}  last_reward={total_reward}")
# Save Q-table

np.save("q_table_two_passenger.npy", Q)
print("\nTraining complete. Q‐table saved as q_table_two_passenger.npy.")