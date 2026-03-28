import random
import time
from envs.maze_env import MazeEnv
from algorithms.monte_carlo_evaluation import BaseMazeModelFreeAlgorithm

class QLearningControl(BaseMazeModelFreeAlgorithm):
    """
    Q-Learning Control with epsilon-greedy policy for exploration.
    An off-policy TD control algorithm.
    """
    def __init__(self, env: MazeEnv, gamma: float = 0.99, alpha: float = 0.1, 
                epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.995):
        super().__init__(env, gamma)
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table: Q(s, a) = 0.0 for all states and actions
        self.Q = {
            (r, c): {a: 0.0 for a in self.actions} 
            for r in range(self.env.height) 
            for c in range(self.env.width)
        }

    def get_epsilon_greedy_action(self, state):
        """
        Returns an action using an epsilon-greedy policy based on current Q-values.
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)
            
        q_values = self.Q[state]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        
        return random.choice(best_actions)

    def train(self, num_episodes: int = 1000, max_steps: int = 1000, verbose: bool = True):
        """
        Trains the agent to find the optimal policy using Q-Learning.
        
        :param num_episodes: How many episodes to train for.
        :param max_steps: Maximum steps allowed per episode.
        :param verbose: Print timing and progress.
        :return: A tuple of (derived_greedy_policy, Action-Value table Q)
        """
        start_time = time.time()
        goals_reached = 0
        
        # Track metrics for visualization
        self.episode_rewards = []
        self.episode_lengths = []
        self.paths = []
        self.Q_history = []
        self.epsilon_history = []
        
        for ep in range(1, num_episodes + 1):
            state = self.env.reset()
            
            path = [state]
            total_reward = 0.0
            steps = 0
            
            for _ in range(max_steps):
                action = self.get_epsilon_greedy_action(state)
                next_state, reward, done = self.env.step(action)
                
                # Q-Learning Update: Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]
                max_next_q = max(self.Q[next_state].values())
                td_target = reward + self.gamma * max_next_q
                td_error = td_target - self.Q[state][action]
                
                self.Q[state][action] += self.alpha * td_error
                
                state = next_state
                
                path.append(state)
                total_reward += reward
                steps += 1
                
                if done:
                    if reward == self.env.reward_goal:
                        goals_reached += 1
                    break
                    
            self.paths.append(path)
            self.episode_lengths.append(steps)
            self.episode_rewards.append(total_reward)
            
            # Decay epsilon after each episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon)
            
            self.epsilon_history.append(self.epsilon)
                
            # Store intermediate Q-values for visualization
            q_copy = {s: {a: q for a, q in q_vals.items()} for s, q_vals in self.Q.items()}
            self.Q_history.append(q_copy)

        if verbose:
            elapsed_time = time.time() - start_time
            print(f"Q-Learning Control completed {num_episodes} episodes in {elapsed_time:.4f} seconds.")
            print(f"Goal reached {goals_reached} times ({(goals_reached / num_episodes) * 100:.2f}%).")
            print(f"Final epsilon: {self.epsilon:.4f}")
            
        return self.get_greedy_policy(), self.Q
        
    def get_greedy_policy(self):
        """
        Extracts the purely deterministic greedy policy from the learned Q-values.
        """
        policy = {}
        for r in range(self.env.height):
            for c in range(self.env.width):
                state = (r, c)
                q_values = self.Q[state]
                max_q = max(q_values.values())
                best_actions = [a for a, q in q_values.items() if q == max_q]
                policy[state] = best_actions[0]
                
        return policy
