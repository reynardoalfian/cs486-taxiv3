import gymnasium as gym

# Load the Taxi-v3 environment with rendering enabled
# Use render_mode="human" to see the visual output
# Use render_mode=None if you don't need the visualization yet
env = gym.make('Taxi-v3', render_mode="human")

# Reset the environment to get the initial state (observation)
observation, info = env.reset()

print(f"Initial Observation: {observation}")
print("Environment setup complete. Let's run a few random steps.")

# Run a few steps with random actions
for step in range(10):
    # Render the current state (if render_mode="human")
    env.render()

    # Sample a random action from the environment's action space
    action = env.action_space.sample() # 0:down, 1:up, 2:right, 3:left, 4:pickup, 5:dropoff

    # Take the action and get the results
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"\nStep: {step + 1}")
    print(f"Action taken: {action}")
    print(f"New Observation: {observation}")
    print(f"Reward received: {reward}")
    print(f"Terminated (goal reached?): {terminated}")
    print(f"Truncated (max steps reached?): {truncated}")

    # Check if the episode has ended
    if terminated or truncated:
        print("\nEpisode finished!")
        # Reset for a new episode if you want to continue
        # observation, info = env.reset()
        break

# Close the environment rendering window
env.close()

print("\nTaxi-v3 environment walkthrough finished.")