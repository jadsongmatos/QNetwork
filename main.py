import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.FloatTensor(state))
        return torch.argmax(act_values).item()  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state)
                target = (reward + self.gamma *
                          torch.max(self.model(next_state)).item())
            state = torch.FloatTensor(state)
            target_f = self.model(state)
            target_f[action] = target
            loss = nn.MSELoss()(self.model(state), target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Define the environment
class Environment:
    def __init__(self, size):
        self.state = np.random.choice([0, 1], size)  # random initialization
        self.position = size // 2

    def step(self, action):
        if action == 0:
            self.position = max(0, self.position - 1)
        else:
            self.position = min(len(self.state) - 1, self.position + 1)

        done = self.check_sequence()
        reward = 1 if done else -1
        return np.copy(self.state), reward, done

    def check_sequence(self):
        # Check if the sequence 010 is found
        if self.position < len(self.state) - 2:
            return list(self.state[self.position:self.position + 3]) == [0, 1, 0]
        return False

    def reset(self):
        self.state = np.random.choice([0, 1], len(self.state))
        self.position = len(self.state) // 2
        return np.copy(self.state)


# Initialize parameters
if __name__ == "__main__":
    n_episodes = 500
    max_timesteps = 100
    batch_size = 64

    # Create the environment and agent
    state_size = 10
    action_size = 2
    env = Environment(state_size)
    agent = DQNAgent(state_size, action_size)
    print(env.state)

    # Training loop
    for e in range(n_episodes):
        state = env.reset()
        total_reward = 0
        for t in range(max_timesteps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode: {e+1}/{n_episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    # Testing loop
    n_test_episodes = 100
    success_count = 0
    for e in range(n_test_episodes):
        state = env.reset()
        for t in range(max_timesteps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            print("step:",next_state, reward, done)
            state = next_state
            if done:
                success_count += 1
                break

    print(f"Agent succeeded in {success_count}/{n_test_episodes} test episodes")
