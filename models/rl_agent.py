"""
rl_agent.py - Reinforcement Learning Agent for Process Scheduling
Uses Q-Learning and Deep Q-Networks (DQN) to learn optimal scheduling policies
"""

import numpy as np
import pickle
import os
from collections import deque, defaultdict
import random


class QLearningAgent:
    """
    Q-Learning agent for process scheduling
    Learns optimal scheduling decisions through trial and error
    """
    
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Q-Learning agent
        
        Args:
            actions: List of possible actions (scheduling decisions)
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
        """
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: stores Q-values for state-action pairs
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        
        # Training statistics
        self.training_rewards = []
        self.training_steps = 0
    
    def get_state_key(self, state):
        """
        Convert state to hashable key for Q-table
        State includes: queue_length, avg_burst, priorities, etc.
        """
        return tuple(state) if isinstance(state, (list, np.ndarray)) else state
    
    def choose_action(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current system state
            training: If True, use exploration; if False, use exploitation only
        
        Returns:
            action_index: Index of chosen action
        """
        state_key = self.get_state_key(state)
        
        # Exploration vs Exploitation
        if training and random.random() < self.epsilon:
            # Explore: choose random action
            return random.randint(0, len(self.actions) - 1)
        else:
            # Exploit: choose best action based on Q-values
            q_values = self.q_table[state_key]
            return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-values using Q-learning update rule
        
        Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
        
        self.training_steps += 1
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath='models/rl_agent.pkl'):
        """Save Q-table and parameters"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'training_rewards': self.training_rewards
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 RL agent saved to {filepath}")
    
    def load(self, filepath='models/rl_agent.pkl'):
        """Load Q-table and parameters"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), data['q_table'])
            self.epsilon = data['epsilon']
            self.training_steps = data['training_steps']
            self.training_rewards = data['training_rewards']
            
            print(f"✅ RL agent loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"❌ Agent file not found: {filepath}")
            return False


class DQNAgent:
    """
    Deep Q-Network agent for process scheduling
    Uses neural network to approximate Q-values for large state spaces
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        """
        Initialize DQN agent
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for neural network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        
        # Experience replay memory
        self.memory = deque(maxlen=10000)
        
        # Build neural network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.training_steps = 0
    
    def _build_model(self):
        """Build neural network for Q-value approximation"""
        try:
            from tensorflow import keras
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            model = Sequential([
                Dense(128, input_dim=self.state_size, activation='relu'),
                Dropout(0.2),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dense(self.action_size, activation='linear')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse'
            )
            
            return model
        except ImportError:
            print("⚠️  TensorFlow not available. Using simple Q-learning instead.")
            return None
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        if self.model is None:
            return random.randint(0, self.action_size - 1)
        
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train network on batch of experiences"""
        if len(self.memory) < self.batch_size or self.model is None:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        # Predict Q-values for current states
        target = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        target_next = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.max(target_next[i])
        
        # Train the model
        self.model.fit(states, target, epochs=1, verbose=0)
        
        self.training_steps += 1
    
    def update_target_model(self):
        """Copy weights from model to target model"""
        if self.model is not None and self.target_model is not None:
            self.target_model.set_weights(self.model.get_weights())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath='models/dqn_agent'):
        """Save model and parameters"""
        if self.model is None:
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(f"{filepath}_model.h5")
        
        data = {
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }
        
        with open(f"{filepath}_params.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 DQN agent saved to {filepath}")
    
    def load(self, filepath='models/dqn_agent'):
        """Load model and parameters"""
        try:
            from tensorflow.keras.models import load_model as keras_load_model
            
            self.model = keras_load_model(f"{filepath}_model.h5")
            self.target_model = keras_load_model(f"{filepath}_model.h5")
            
            with open(f"{filepath}_params.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.epsilon = data['epsilon']
            self.training_steps = data['training_steps']
            
            print(f"✅ DQN agent loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading DQN agent: {e}")
            return False


# Demo and training functions
def train_rl_agent_demo(episodes=1000):
    """
    Demo: Train RL agent on synthetic scheduling environment
    """
    print("=" * 70)
    print("🚀 Training RL Agent for Process Scheduling")
    print("=" * 70)
    
    # Define actions (scheduling algorithms to choose from)
    actions = ['FCFS', 'SJF', 'PRIORITY', 'RR']
    
    # Initialize agent
    agent = QLearningAgent(actions=actions)
    
    print(f"\n🤖 Initialized Q-Learning agent")
    print(f"   Actions: {actions}")
    print(f"   Learning rate: {agent.learning_rate}")
    print(f"   Discount factor: {agent.discount_factor}")
    
    # Training loop
    print(f"\n🏋️  Training for {episodes} episodes...")
    
    for episode in range(episodes):
        # Simulate scheduling environment
        # State: [queue_length, avg_burst_time, priority_range, cpu_load]
        state = [
            random.randint(1, 20),  # queue_length
            random.randint(5, 50),  # avg_burst_time
            random.randint(1, 10),  # priority_range
            random.randint(20, 100) # cpu_load
        ]
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 10:
            # Choose action
            action = agent.choose_action(state)
            
            # Simulate environment response
            # Reward based on chosen algorithm's performance
            reward = simulate_scheduling_reward(state, actions[action])
            
            # Next state (environment changes)
            next_state = [
                max(1, state[0] + random.randint(-3, 3)),
                max(5, state[1] + random.randint(-10, 10)),
                max(1, state[2] + random.randint(-2, 2)),
                max(20, state[3] + random.randint(-15, 15))
            ]
            
            # Update Q-values
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps >= 10:
                done = True
        
        agent.training_rewards.append(total_reward)
        agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(agent.training_rewards[-100:])
            print(f"   Episode {episode + 1}/{episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # Save trained agent
    agent.save()
    
    print(f"\n✅ Training complete!")
    print(f"   Total episodes: {episodes}")
    print(f"   Final epsilon: {agent.epsilon:.3f}")
    print(f"   Q-table size: {len(agent.q_table)}")
    
    return agent


def simulate_scheduling_reward(state, algorithm):
    """
    Simulate reward for choosing a scheduling algorithm
    Higher reward for better performance in given state
    """
    queue_len, avg_burst, priority_range, cpu_load = state
    
    # Different algorithms perform better in different scenarios
    if algorithm == 'FCFS':
        reward = 10 - queue_len * 0.5
    elif algorithm == 'SJF':
        reward = 15 - avg_burst * 0.2 + (100 - cpu_load) * 0.1
    elif algorithm == 'PRIORITY':
        reward = 12 + priority_range * 0.5
    elif algorithm == 'RR':
        reward = 13 - (avg_burst - 20) * 0.1
    else:
        reward = 8
    
    # Add some noise
    reward += random.uniform(-2, 2)
    
    return reward


if __name__ == "__main__":
    # Train RL agent
    trained_agent = train_rl_agent_demo(episodes=500)
    
    print("\n" + "=" * 70)
    print("✨ RL Agent ready for intelligent scheduling decisions!")
    print("=" * 70)