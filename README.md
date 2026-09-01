# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement

Implement the Q-Learning control algorithm using the Gymnasium FrozenLake-v1 environment. The objective is to train an agent to learn the optimal action-value function and find a suitable path from the starting state to the goal while avoiding holes. The agent should use an epsilon-greedy strategy to balance exploration and exploitation.



## Software Requirements
Programming Language: Python 3
Platform: Google Colab / Jupyter Notebook
Libraries:
Gymnasium – for the FrozenLake environment
NumPy – for Q-table and numerical calculations
Matplotlib – for plotting the learning curve
Algorithm: Q-Learning
Environment: FrozenLake-v1
Hardware: Basic computer with internet access
Python packages:



## Environment Description

FrozenLake-v1 is a grid-world reinforcement learning environment consisting of a 4 × 4 grid. The agent starts from the initial state and must reach the goal while avoiding holes.

States: 16
Actions: 4
0 – Left
1 – Down
2 – Right
3 – Up
Reward:
+1 for reaching the goal
0 for other transitions, including falling into a hole
Episode termination: The episode ends when the agent reaches the goal or falls into a hole.
Slippery environment: is_slippery=True, so the agent's movement can be stochastic.
Learning: The Q-table is updated using the Q-Learning update rule, while epsilon-greedy action selection allows the agent to explore and exploit.


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm



## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
```

```python
# -------------------------------------------------
# Create Grid Environment
# -------------------------------------------------

class GridWorld(gym.Env):

    def __init__(self):
        super().__init__()

        # 4 x 4 grid
        self.rows = 4
        self.cols = 4

        # Start and Goal
        self.start = (0, 0)
        self.goal = (3, 3)

        # Obstacles
        self.obstacles = {
            (1, 1),
            (1, 2),
            (2, 1)
        }

        # 4 actions:
        # 0 = Left
        # 1 = Down
        # 2 = Right
        # 3 = Up

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(16)

        self.state = self.start

    def state_to_number(self, state):
        return state[0] * self.cols + state[1]

    def number_to_state(self, state):
        return (state // self.cols, state % self.cols)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = self.start

        return self.state_to_number(self.state), {}

    def step(self, action):

        row, col = self.state

        # Movement
        if action == 0:       # Left
            new_row, new_col = row, col - 1

        elif action == 1:     # Down
            new_row, new_col = row + 1, col

        elif action == 2:     # Right
            new_row, new_col = row, col + 1

        else:                 # Up
            new_row, new_col = row - 1, col

        # Check boundaries
        if (
            new_row < 0 or
            new_row >= self.rows or
            new_col < 0 or
            new_col >= self.cols
        ):
            new_row, new_col = row, col
            reward = -2

        # Check obstacles
        elif (new_row, new_col) in self.obstacles:
            new_row, new_col = row, col
            reward = -2

        else:
            reward = -1

        self.state = (new_row, new_col)

        # Goal
        terminated = self.state == self.goal

        if terminated:
            reward = 10

        truncated = False

        return (
            self.state_to_number(self.state),
            reward,
            terminated,
            truncated,
            {}
        )


env = GridWorld()


```

```python
# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

learning_rate = 0.8
discount_factor = 0.95

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 5000
```

```python
# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((
    env.observation_space.n,
    env.action_space.n
))


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def choose_action(state):

    # Exploration
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation
    return np.argmax(Q[state])


```

```python
# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

episode_rewards = []

for episode in range(episodes):

    state, info = env.reset()

    total_reward = 0
    done = False

    while not done:

        # Choose action
        action = choose_action(state)

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Q-Learning update
        best_next_value = np.max(Q[next_state])

        Q[state, action] = Q[state, action] + learning_rate * (
            reward
            + discount_factor * best_next_value
            - Q[state, action]
        )

        state = next_state

        total_reward += reward

    episode_rewards.append(total_reward)

    # Reduce exploration
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )



```

```python
# -------------------------------------------------
# Calculate Value Function and Policy
# -------------------------------------------------

state_values = np.max(Q, axis=1)

learned_policy = np.argmax(Q, axis=1)
```

```python
# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)

```

```python
# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", average_reward)
```

```python
# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Q-Learning Curve - Grid World")
plt.grid(True)
plt.show()

env.close()
```
---

## Output

<img width="527" height="670" alt="Screenshot 2026-09-01 155821" src="https://github.com/user-attachments/assets/d8d16f30-42ae-41d8-922b-78bde46de098" />

<img width="914" height="591" alt="Screenshot 2026-09-01 155827" src="https://github.com/user-attachments/assets/7d353466-b349-4e5b-b2cf-10f20247c8b7" />

---

## Result
Q-Learning successfully learned a policy for reaching the goal in FrozenLake while avoiding holes.

---

## Inference

The experiment demonstrates that Q-Learning can learn an effective policy through trial and error without being given the optimal path beforehand. By balancing exploration and exploitation using epsilon-greedy action selection, the agent gradually learns better actions for reaching the goal while avoiding holes. Thus, Q-Learning is effective for solving the FrozenLake reinforcement learning problem.






---

