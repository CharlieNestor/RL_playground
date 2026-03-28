import random
import time
from envs.maze_env import MazeEnv
from algorithms.monte_carlo_evaluation import BaseMazeModelFreeAlgorithm

class MonteCarloControl(BaseMazeModelFreeAlgorithm):
    """
    Monte Carlo Control with epsilon-greedy policy.
    Learns the optimal policy by maintaining an action-value function Q(s, a).
    """
    def __init__(self, env: MazeEnv, gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.995):
        super().__init__(env, gamma)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table: Q(s, a) = 0.0 for all states and actions
        self.Q = {
            (r, c): {a: 0.0 for a in self.actions} 
            for r in range(self.env.height) 
            for c in range(self.env.width)
        }
        
        # Counter for incremental mean: N(s, a) = number of times action 'a' was taken in state 's'
        self.N = {
            (r, c): {a: 0 for a in self.actions}
            for r in range(self.env.height)
            for c in range(self.env.width)
        }

    def get_epsilon_greedy_action(self, state):
        """
        Returns an action using an epsilon-greedy policy based on the current Q-values.
        """
        # Exploration: choose a random action with probability epsilon (ε)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
            
        # Exploitation: choose the action with the maximum Q-value with probability 1 - ε
        # If multiple actions have the same max Q-value, break ties randomly
        q_values = self.Q[state]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        
        return random.choice(best_actions)

    def generate_episode(self, max_steps: int = 1000) -> list:
        """
        Generates a single episode strictly following the current epsilon-greedy policy.
        """
        episode = []
        state = self.env.reset()
        
        for _ in range(max_steps):
            action = self.get_epsilon_greedy_action(state)
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward, next_state))
            
            if done:
                break
            state = next_state
            
        return episode

    def train(self, num_episodes: int = 1000, max_steps: int = 1000, first_visit: bool = True, verbose: bool = True):
        """
        Trains the agent to find the optimal policy using Monte Carlo Control.
        We are implementing Generalized Policy Iteration with MC:
        We don't evaluate the policy up until convergence. We take one step of policy evaluation 
        (evaluating the policy for just ONE episode) and then immediately improve it.
        
        :param num_episodes: How many episodes to train for.
        :param max_steps: Maximum steps allowed per episode.
        :param first_visit: If True, uses First-Visit MC. If False, Every-Visit.
        :param verbose: Print timing and progress.
        :return: A tuple of (derived_greedy_policy, Action-Value table Q)
        """
        start_time = time.time()
        # Counter for how many episodes ended by reaching the goal
        goals_reached = 0
        
        # Track metrics for visualization
        self.episode_rewards = []
        self.episode_lengths = []
        self.paths = []
        self.Q_history = []
        self.epsilon_history = []
        
        for ep in range(1, num_episodes + 1):
            episode = self.generate_episode(max_steps=max_steps)
            
            # Store visualization metrics
            path = [step[0] for step in episode]
            if len(episode) > 0:
                # Add the final state reached in the episode
                path.append(episode[-1][3])
            
            self.paths.append(path)
            self.episode_lengths.append(len(episode))
            
            total_reward = sum([step[2] for step in episode])
            self.episode_rewards.append(total_reward)
            
            # Check if the episode ended by reaching the goal
            if len(episode) > 0 and episode[-1][2] == self.env.reward_goal:
                goals_reached += 1
                
            G = 0.0
            # Track visited state-action pairs for first-visit MC
            state_action_pairs = [(x[0], x[1]) for x in episode]
            
            # Iterate backwards through the episode
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward, _ = episode[t]
                G = self.gamma * G + reward
                
                # First-Visit MC Check
                if first_visit and (state, action) in state_action_pairs[:t]:
                    continue
                    
                # Increment count for (state, action) pair
                self.N[state][action] += 1
                
                # Incremental Mean Update Formula for Q(s, a):
                # Q(S_t, A_t) <- Q(S_t, A_t) + (1 / N(S_t, A_t)) * [G_t - Q(S_t, A_t)]
                alpha = 1.0 / self.N[state][action]
                self.Q[state][action] = self.Q[state][action] + alpha * (G - self.Q[state][action])
            
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
            print(f"Monte Carlo Control completed {num_episodes} episodes in {elapsed_time:.4f} seconds.")
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
                # Just pick one optimal action if there are ties
                best_actions = [a for a, q in q_values.items() if q == max_q]
                # In control, typically we just store a deterministic dictionary {state: action}
                # or a 100% prob for the best action {state: {action: 1.0}}
                policy[state] = best_actions[0]
                
        return policy
