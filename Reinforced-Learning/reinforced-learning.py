# Importing necessary libraries
import gym  # OpenAI Gym provides a variety of environments to simulate reinforcement learning problems
import numpy as np  # NumPy is used for array handling and mathematical operations
import random  # Random is used to generate random numbers
import matplotlib.pyplot as plt  # Matplotlib is used for plotting graphs

# Initialize the Frozen Lake environment from OpenAI Gym
env = gym.make("FrozenLake-v1", is_slippery=True)

# Q-table initialization:
# The Q-table stores Q-values, which represent the expected future rewards for taking a specific action in a given state.
# The table is initialized with zeros. Rows represent states, and columns represent actions.
action_space_size = env.action_space.n  # Number of possible actions (left, down, right, up)
state_space_size = env.observation_space.n  # Number of possible states
q_table = np.zeros((state_space_size, action_space_size))  # Initialize Q-table with zeros

# Hyperparameters: These control the learning process.
alpha = 0.1  # Learning rate: Determines how much new information overrides the old.
gamma = 0.99  # Discount factor: Balances immediate and future rewards.
epsilon = 1.0  # Exploration rate: Controls how much the agent explores versus exploits.
epsilon_decay = 0.995  # Decay rate for epsilon: Reduces exploration over time.
min_epsilon = 0.01  # Minimum exploration rate: Ensures some exploration happens.
episodes = 10000  # Number of episodes to train the agent.
max_steps = 100  # Max steps per episode: Controls how long each episode can last.

# List to keep track of rewards for analysis
all_rewards = []

# Q-Learning Algorithm
for episode in range(episodes):
    state = env.reset()  # Reset the environment to start a new episode, getting the initial state
    total_rewards = 0  # Keep track of total rewards earned in this episode

    for step in range(max_steps):
        # Exploration-exploitation trade-off:
        # The agent chooses either to explore new actions or exploit known actions based on the epsilon value.
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore: Randomly choose an action
        else:
            action = np.argmax(q_table[state])  # Exploit: Choose the action with the highest Q-value for the current state

        # Take the chosen action and observe the outcome
        new_state, reward, done, _ = env.step(action)

        # Q-learning update rule:
        # Update the Q-value for the current state-action pair using the Bellman equation.
        # New Q-value = Old Q-value + learning rate * (reward + discount factor * max future Q-value - old Q-value)
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[new_state]) - q_table[state, action]
        )

        # Transition to the next state
        state = new_state

        # Accumulate the reward for this step
        total_rewards += reward

        # If the episode ends (either by reaching the goal or falling into a hole), stop the episode
        if done:
            break

    # Decay epsilon to reduce exploration over time, favoring exploitation as the agent learns more
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Append the total reward of this episode to our list for later analysis
    all_rewards.append(total_rewards)

print("Training completed.\n")

# Evaluating the trained agent
# This section tests how well the trained agent performs by running 100 episodes and averaging the rewards.
total_rewards = 0  # Reset total rewards for evaluation

for episode in range(100):
    state = env.reset()  # Start a new episode
    done = False  # Flag to check if the episode has ended

    for step in range(max_steps):
        action = np.argmax(q_table[state])  # Choose the best action based on the Q-table (no exploration)
        new_state, reward, done, _ = env.step(action)  # Take the action and observe the outcome
        state = new_state  # Move to the next state
        total_rewards += reward  # Accumulate the rewards

        if done:  # If the episode ends, stop this loop
            break

# Calculate and print the average reward over 100 evaluation episodes
print(f"Average reward over 100 episodes: {total_rewards / 100}")

# Plotting the rewards over episodes
# This graph shows how rewards change as training progresses, which can indicate learning.
plt.plot(range(episodes), all_rewards)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rewards over Time')
plt.show()
