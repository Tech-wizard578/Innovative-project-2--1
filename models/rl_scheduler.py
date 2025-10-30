import numpy as np
import pickle

class RLScheduler:
    def __init__(self, n_states=100, n_actions=10, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states    # discretization of system state
        self.n_actions = n_actions  # number of possible jobs to pick
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate

        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def learn(self, state, action, reward, next_state):
        # Q-Learning update
        predict = self.Q[state, action]
        target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - predict)

    def train(self, env, n_episodes=500):
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state

    def save_model(self, path="models/rl_q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)

    def load_model(self, path="models/rl_q_table.pkl"):
        with open(path, "rb") as f:
            self.Q = pickle.load(f)
