import random
import time
import math
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from envs.maze_env import MazeEnv
from algorithms.monte_carlo_evaluation import BaseMazeModelFreeAlgorithm

class DQNNetwork(nn.Module):
    """
    A simple Feed-Forward Neural Network to approximate Q-values.
    
    Architecture:
    - Input: A one-hot encoded vector representing the agent's current grid position.
            Size = height * width
    - Hidden Layers: Two Linear layers with ReLU activations.
    - Output: A Linear layer outputting Q-values for each possible action.
            Size = 4 (UP, DOWN, LEFT, RIGHT).
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes forward pass through the network.
        :param x: Batch of One-Hot Encoded States.
        :return: Q-Values batch for all actions.
        """
        x = F.relu(self.fc1(x))     # apply ReLU activation function to output of first fully connected layer
        x = F.relu(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    """
    Experience Replay Buffer to store past transitions.
    
    Why it is needed?
    1. Breaks temporal correlation: Consecutive states are highly correlated in RL, which
       unbalances network training. Sampling randomly from memory prevents this.
    2. Data Efficiency: We can reuse the same past transitions multiple times.
    """
    def __init__(self, capacity: int):
        # deque automatically discards oldest items when capacity is reached
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Randomly samples a batch of transitions for training."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNControl(BaseMazeModelFreeAlgorithm):
    """
    Deep Q-Network (DQN) Control Algorithm.
    
    Instead of using a tabular Q(S, A) dictionary, this algorithm predicts Q-values
    using a Feed-Forward Neural Network trained via Gradient Descent.
    
    Key Instruments Introduced:
    - Replay Buffer: Stores and samples uncorrelated past transitions.
    - Target Network: A cloned static network to provide stable Bellman targets.
    - Neural Network Policy: Approximates the Q-value function.
    """
    def __init__(self, env: MazeEnv, gamma: float = 0.99, lr: float = 1e-3, 
                 epsilon_start: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 memory_capacity: int = 10000, batch_size: int = 32, target_update_freq: int = 10,
                 hidden_size: int = 64):
        """
        Initializes the DQN Agent.
        
        :param env: The environment the agent will interact with.
        :param gamma: Discount factor for future rewards (0 to 1).
        :param lr: Learning rate for the neural network optimizer.
        :param epsilon_start: Initial exploration rate for epsilon-greedy action selection.
        :param epsilon_min: Minimum bound for the exploration rate.
        :param epsilon_decay: Multiplier to decay epsilon after each episode.
        :param memory_capacity: Maximum size of the Experience Replay Buffer.
        :param batch_size: Number of transitions sampled from the buffer for each training step.
        :param target_update_freq: Frequency (in episodes) to synchronize the Target Network with the Policy Network.
        :param hidden_size: Number of neurons in the hidden layers of the Policy and Target networks.
        """
        super().__init__(env, gamma)
        
        # Hyperparameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.hidden_size = hidden_size
        
        # Determine sizes for our Neural Network Environment
        self.n_states = self.env.height * self.env.width
        self.n_actions = len(self.actions)
        
        # Instrument: Policy and Target Networks
        # 1. Policy Network: This is the active network. We train its weights at every step
        #    using gradient descent to approximate the optimal Q-value function Q(S,A).
        self.policy_net = DQNNetwork(input_size=self.n_states, hidden_size=self.hidden_size, output_size=self.n_actions)
        
        # 2. Target Network: A "frozen" copy of the Policy Network.
        #    Why? In standard Q-Learning, the target Q-value (R + gamma * max_a(Q(S', a))) 
        #    changes at every step while we update Q(S, A), causing instability ("chasing a moving target").
        #    We compute the target using this frozen network, and only periodically copy
        #    the weights from the Policy network, drastically stabilizing training.
        self.target_net = DQNNetwork(input_size=self.n_states, hidden_size=self.hidden_size, output_size=self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())   # copy weights from policy_net to target_net
        self.target_net.eval()  # Set to evaluation mode
        
        # Instrument: Optimizer setup
        # Adam Optimizer automatically tunes learning rates based on gradients
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Instrument: Experience Replay
        self.memory = ReplayBuffer(capacity=memory_capacity)

    def _state_to_tensor(self, state) -> torch.Tensor:
        """
        Converts a scalar/tuple state like (row, col) into a One-Hot Encoded Vector, 
        suitable for our Feed-Forward Neural Network input.
        """
        r, c = state
        # Flattened grid index
        state_idx = r * self.env.width + c
        
        # Create a float tensor of all zeros, and encode a 1.0 at the current state index
        state_tensor = torch.zeros(self.n_states, dtype=torch.float32)
        state_tensor[state_idx] = 1.0
        return state_tensor

    def get_epsilon_greedy_action(self, state_tensor: torch.Tensor):
        """
        Selects an Action based on the Epsilon-Greedy strategy, probing the neural network.
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)
            
        with torch.no_grad(): # We don't want to compute gradients during action selection
            # Forward pass: policy_net outputs 4 Q-values. We take the argmax index.
            q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values).item())

    def _optimize_model(self):
        """
        Samples a batch from the Replay Buffer and updates the policy network via Backpropagation.
        """
        # If memory has fewer samples than the batch size, we don't start training yet
        if len(self.memory) < self.batch_size:
            return
            
        # Sample mini-batch randomly
        transitions = self.memory.sample(self.batch_size)
        
        # Unpack the batch
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        
        # Stack the individual tensors into batch tensors -> Shape: [batch_size, n_states]
        # state_batch and next_state_batch are already tensors from _state_to_tensor
        state_batch = torch.stack(batch_state)
        next_state_batch = torch.stack(batch_next_state)
        
        # action_batch, reward_batch and done_batch are lists, we need to convert them to tensors
        action_batch = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(1)    # Shape: [batch_size, 1]
        reward_batch = torch.tensor(batch_reward, dtype=torch.float32)               # Shape: [batch_size]
        done_batch = torch.tensor(batch_done, dtype=torch.float32)                   # Shape: [batch_size]
        
        # 1. Compute Predicted Q-Values: Q(S, A)
        # policy_net(state_batch) outputs Size=[batch_size, 4].
        # gather(1, action_batch) picks the specific Q-value of the action we actually took
        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        
        # 2. Compute Target Q-Values: R + gamma * max_a(TargetNet(S'))
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            
            # If done == 1.0, the episode terminated, there is no next state Q-value.
            # torch.where acts as an explicit if-else for each element in the batch:
            # if done_batch == 1.0 -> expected_state_action_values = reward_batch
            # else -> expected_state_action_values = reward_batch + gamma * next_state_values
            expected_state_action_values = torch.where(
                done_batch == 1.0, 
                reward_batch, 
                reward_batch + (self.gamma * next_state_values)
            )
            
        # 3. Compute Loss
        # MSE is standard tabular Q-Learning. Huber loss is smoother for DL outliers.
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        
        # 4. Perform Gradient Descent (Backpropagation)
        self.optimizer.zero_grad() # Clear out accumulated gradients
        loss.backward()            # Computes d(Loss)/d(Weights)
        self.optimizer.step()      # Updates the weights using Adam
        
    def train(self, num_episodes: int = 1000, max_steps: int = 1000, verbose: bool = True):
        """
        Trains the DQN agent over multiple episodic runs.
        """
        start_time = time.time()
        goals_reached = 0
        
        # Tracking metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.paths = []
        self.epsilon_history = []
        self.Q_history = []
        
        for ep in range(1, num_episodes + 1):
            state = self.env.reset()
            state_tensor = self._state_to_tensor(state)
            
            path = [state]
            total_reward = 0.0
            steps = 0
            
            for _ in range(max_steps):
                action = self.get_epsilon_greedy_action(state_tensor)
                next_state, reward, done = self.env.step(action)
                next_state_tensor = self._state_to_tensor(next_state)
                
                # Save the transition into Experience Replay Buffer
                self.memory.push(state_tensor, action, reward, next_state_tensor, done)
                
                # Perform one step of Network Optimization via Gradient Descent
                self._optimize_model()
                
                state_tensor = next_state_tensor
                state = next_state # tracking for visual purposes
                
                path.append(state)
                total_reward += reward
                steps += 1
                
                if done:
                    if reward == self.env.reward_goal:
                        goals_reached += 1
                    break
                    
            # End of Episode
            self.paths.append(path)
            self.episode_lengths.append(steps)
            self.episode_rewards.append(total_reward)
            self.epsilon_history.append(self.epsilon)
            
            # Update target network occasionally
            if ep % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            # Decay Epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon)

            # Store intermediate Q-values for visualization
            self.Q_history.append(self._extract_q_values())

        if verbose:
            elapsed_time = time.time() - start_time
            print(f"DQN Control completed {num_episodes} episodes in {elapsed_time:.4f} seconds.")
            print(f"Goal reached {goals_reached} times ({(goals_reached / num_episodes) * 100:.2f}%).")
            print(f"Final epsilon: {self.epsilon:.4f}")
            
        return self.get_greedy_policy(), self.policy_net
        
    def get_greedy_policy(self):
        """
        Extracts a deterministic dictionary policy {state: action} from the learned Neural Net Q-Values.
        """
        policy = {}
        with torch.no_grad():
            for r in range(self.env.height):
                for c in range(self.env.width):
                    state = (r, c)
                    # We can't query if it is a wall right now, so we map all grids just like Tabular
                    state_tensor = self._state_to_tensor(state)
                    q_values = self.policy_net(state_tensor)
                    best_action = int(torch.argmax(q_values).item())
                    policy[state] = best_action
        return policy

    def _extract_q_values(self):
        """
        Extracts Q-values for all states to match the tabular representation {state: {action: value}}.
        Useful for visualization and tracking history episode by episode.
        """
        q_dict = {}
        with torch.no_grad():
            for r in range(self.env.height):
                for c in range(self.env.width):
                    state = (r, c)
                    state_tensor = self._state_to_tensor(state)
                    # policy_net outputs a tensor of shape [4], convert to numpy array to extract floats
                    q_vals = self.policy_net(state_tensor).numpy()
                    q_dict[state] = {a: float(q_vals[a]) for a in self.actions}
        return q_dict

