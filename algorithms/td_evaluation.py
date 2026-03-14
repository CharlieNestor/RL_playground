import random
import time
from envs.maze_env import MazeEnv, UP, DOWN, LEFT, RIGHT
from algorithms.monte_carlo_evaluation import BaseMazeModelFreeAlgorithm

class TDPrediction(BaseMazeModelFreeAlgorithm):
    """
    Temporal Difference (TD(0)) Policy Evaluation.
    Estimates the state-value function V(s) for a given policy by bootstrapping
    from the current estimates of next states. 
    """
    def __init__(self, env: MazeEnv, gamma: float = 0.99, alpha: float = 0.1):
        super().__init__(env, gamma)
        self.alpha = alpha
        
        # Initialize Value table: V(s) = 0 for all states
        self.V = {
            (r, c): 0.0 
            for r in range(self.env.height) 
            for c in range(self.env.width)
        }

    def evaluate_policy(self, policy: dict, num_episodes: int = 1000, max_steps: int = 100, verbose: bool = True, true_v: dict = None):
        """
        Evaluates the policy using TD(0).
        
        :param policy: The policy to evaluate.
        :param num_episodes: How many episodes to sample.
        :param max_steps: Maximum steps allowed per episode.
        :param verbose: Print timing and progress.
        :param true_v: Optional ground truth Value table for calculating RMSE.
        :return: Estimated Value table V(s).
        """
        start_time = time.time()
        goals_reached = 0
        self.errors = []
        
        # Initialize lists to store step-by-step history
        self.history_ep1 = []
        self.history_ep2 = []
        self.history_ep3 = []
        self.history_first_success = []
        first_success_recorded = False
        
        for ep in range(1, num_episodes + 1):
            # Restart from initial state at the beg of each episode
            state = self.env.reset()
            current_ep_history = []
            
            for _ in range(max_steps):
                if state not in policy:
                    break
                    
                # Sample an action according to the policy
                action_probs = policy[state]
                actions = list(action_probs.keys())
                probs = list(action_probs.values())
                action = random.choices(actions, weights=probs, k=1)[0]
                
                # Take step
                next_state, reward, done = self.env.step(action)
                
                # TD(0) Update for V(s)
                # TD Error: delta = R + gamma * V(s') - V(s)
                # V(s) <- V(s) + alpha * delta
                td_target = reward + self.gamma * self.V[next_state]
                td_error = td_target - self.V[state]
                
                self.V[state] = self.V[state] + self.alpha * td_error
                current_ep_history.append(self.V.copy())
                
                if done:
                    if reward == self.env.reward_goal:
                        goals_reached += 1
                        if not first_success_recorded:
                            first_success_recorded = True
                            self.history_first_success = current_ep_history
                    break
                state = next_state
                
            if ep == 1:
                self.history_ep1 = current_ep_history
            elif ep == 2:
                self.history_ep2 = current_ep_history
            elif ep == 3:
                self.history_ep3 = current_ep_history
                
            # Calculate RMSE if true_v is provided
            if true_v is not None:
                rmse = (sum((self.V[s] - true_v[s])**2 for s in true_v) / len(true_v)) ** 0.5
                self.errors.append(rmse)
                
        if verbose:
            elapsed_time = time.time() - start_time
            print(f"TD(0) prediction completed {num_episodes} episodes in {elapsed_time:.4f} seconds.")
            print(f"Goal reached {goals_reached} times ({(goals_reached / num_episodes) * 100:.2f}%).")
            
        return self.V


class TDLambdaPrediction(BaseMazeModelFreeAlgorithm):
    """
    Temporal Difference (TD(lambda)) Policy Evaluation.
    Estimates the state-value function V(s) for a given policy using backward view 
    TD approach via eligibility traces.
    TD(lambda) bridges the gap between TD(0) (lambda=0) and Monte Carlo (lambda=1).
    """
    def __init__(self, env: MazeEnv, gamma: float = 0.99, alpha: float = 0.1, lam: float = 0.9):
        super().__init__(env, gamma)
        self.alpha = alpha
        self.lam = lam  # lambda parameter (0 <= lambda <= 1)
        
        # Initialize Value table: V(s) = 0 for all states
        self.V = {
            (r, c): 0.0 
            for r in range(self.env.height) 
            for c in range(self.env.width)
        }

    def evaluate_policy(self, policy: dict, num_episodes: int = 1000, max_steps: int = 100, verbose: bool = True, true_v: dict = None):
        """
        Evaluates the policy using TD(lambda) with eligibility traces.
        
        :param policy: The policy to evaluate.
        :param num_episodes: How many episodes to sample.
        :param max_steps: Maximum steps allowed per episode.
        :param verbose: Print timing and progress.
        :param true_v: Optional ground truth Value table for calculating RMSE.
        :return: Estimated Value table V(s).
        """
        start_time = time.time()
        goals_reached = 0
        self.errors = []
        
        # Initialize lists to store step-by-step history
        self.history_ep1 = []
        self.history_ep2 = []
        self.history_ep3 = []
        self.history_first_success = []
        first_success_recorded = False
        
        for ep in range(1, num_episodes + 1):
            state = self.env.reset()
            current_ep_history = []
            
            # Initialize Eligibility Traces E(s) = 0 for all states for this episode
            E = {
                (r, c): 0.0 
                for r in range(self.env.height) 
                for c in range(self.env.width)
            }
            
            for _ in range(max_steps):
                if state not in policy:
                    break
                    
                # Sample an action according to the policy
                action_probs = policy[state]
                actions = list(action_probs.keys())
                probs = list(action_probs.values())
                action = random.choices(actions, weights=probs, k=1)[0]
                
                # Take step
                next_state, reward, done = self.env.step(action)
                
                # TD Error: delta = R + gamma * V(S') - V(S)
                # Next state V(S') is 0 if it's a terminal state, but our self.V table naturally has 0 for it
                td_error = reward + self.gamma * self.V[next_state] - self.V[state]
                
                # Update the eligibility trace for the current state (Accumulating trace)
                E[state] += 1.0
                
                # Update V(s) and E(s) for ALL states
                for r in range(self.env.height):
                    for c in range(self.env.width):
                        s = (r, c)
                        # V(s) <- V(s) + alpha * delta * E(s)
                        self.V[s] = self.V[s] + self.alpha * td_error * E[s]
                        # Decay the eligibility trace
                        E[s] = self.gamma * self.lam * E[s]
                        
                current_ep_history.append(self.V.copy())
                        
                if done:
                    if reward == self.env.reward_goal:
                        goals_reached += 1
                        if not first_success_recorded:
                            first_success_recorded = True
                            self.history_first_success = current_ep_history
                    break
                state = next_state
                
            if ep == 1:
                self.history_ep1 = current_ep_history
            elif ep == 2:
                self.history_ep2 = current_ep_history
            elif ep == 3:
                self.history_ep3 = current_ep_history
                
            # Calculate RMSE if true_v is provided
            if true_v is not None:
                rmse = (sum((self.V[s] - true_v[s])**2 for s in true_v) / len(true_v)) ** 0.5
                self.errors.append(rmse)
                
        if verbose:
            elapsed_time = time.time() - start_time
            print(f"TD(lambda) prediction completed {num_episodes} episodes in {elapsed_time:.4f} seconds.")
            print(f"Goal reached {goals_reached} times ({(goals_reached / num_episodes) * 100:.2f}%).")
            
        return self.V
