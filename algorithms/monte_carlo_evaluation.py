import random
import time
from envs.maze_env import MazeEnv, UP, DOWN, LEFT, RIGHT

class BaseMazeModelFreeAlgorithm:
    """
    Base class for Model-Free algorithms in a Maze environment.
    Unlike DP algorithms, model-free methods do not have access to a perfect model
    of the environment (no _get_transition). They must learn by actually interacting
    with the environment using env.reset() and env.step(action).
    """
    def __init__(self, env: MazeEnv, gamma: float = 0.99):
        self.env = env
        self.gamma = gamma
        self.actions = self.env.get_actions()

class MonteCarloPrediction(BaseMazeModelFreeAlgorithm):
    """
    Monte Carlo Policy Evaluation (Prediction).
    Estimates the state-value function V(s) for a given policy by averaging
    sampled returns from actual episodes interacting with the environment.
    """
    def __init__(self, env: MazeEnv, gamma: float = 0.99):
        super().__init__(env, gamma)
        # Initialize Value table: V(s) = 0 for all states
        self.V = {
            (r, c): 0.0 
            for r in range(self.env.height) 
            for c in range(self.env.width)
        }
        # Counter for incremental mean: N(s) = number of times state was visited
        self.N = {
            (r, c): 0
            for r in range(self.env.height)
            for c in range(self.env.width)
        }

    def generate_episode(self, policy: dict, max_steps: int = 1000) -> list:
        """
        Generates a single episode strictly following the given policy.
        
        :param policy: Dictionary mapping states to action probabilities.
        :param max_steps: Max steps to prevent infinite loops in bad policies.
        :return: A list of (state, action, reward) tuples representing the episode trajectory.
        """
        episode = []
        state = self.env.reset()
        
        for _ in range(max_steps):
            # If we end up in a state without a defined policy (e.g. wall or goal), we stop.
            if state not in policy:
                break
                
            # Sample an action according to the policy's probabilities
            action_probs = policy[state]
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            action = random.choices(actions, weights=probs, k=1)[0]
            
            # Interact with the environment
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            
            if done:
                break
            state = next_state
            
        return episode
        
    def evaluate_policy(self, policy: dict, num_episodes: int = 1000, max_steps: int = 1000, first_visit: bool = True, verbose: bool = True, true_v: dict = None):
        """
        Evaluates the policy using Monte Carlo.
        
        :param policy: The policy to evaluate.
        :param num_episodes: How many episodes to sample.
        :param max_steps: Maximum steps allowed per episode.
        :param first_visit: If True, uses First-Visit MC. If False, uses Every-Visit MC.
        :param verbose: Print timing and progress.
        :param true_v: Optional ground truth Value table for calculating MSE.
        :return: Estimated Value table V(s).
        """
        start_time = time.time()
        goals_reached = 0
        
        # Initialize lists to store step-by-step history
        self.history_ep1 = []
        self.history_ep2 = []
        self.history_ep3 = []
        self.history_first_success = []
        self.errors = []
        first_success_recorded = False
        
        for ep in range(1, num_episodes + 1):
            episode = self.generate_episode(policy, max_steps=max_steps)
            
            # Check if the episode ended by reaching the goal (reward == reward_goal)
            is_success = False
            if len(episode) > 0 and episode[-1][2] == self.env.reward_goal:
                goals_reached += 1
                is_success = True
                
            is_first_success = is_success and not first_success_recorded
            if is_first_success:
                first_success_recorded = True
                
            G = 0.0
            # List of just the states visited in this episode (for First-Visit check)
            states_in_episode = [x[0] for x in episode]
            
            # Iterate backwards through the episode
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                # First-Visit MC Check
                # If the state was visited earlier in the episode, skip this calculation
                if first_visit and state in states_in_episode[:t]:
                    continue
                    
                # Increment visit counter for this state
                self.N[state] += 1
                
                # Incremental Mean Update Formula:
                # V(s) <- V(s) + (1 / N(s)) * [G - V(s)]
                # Since 1/N(s) decreases over time, this naturally acts as a decaying learning rate.
                alpha = 1.0 / self.N[state]
                self.V[state] = self.V[state] + alpha * (G - self.V[state])
                
                # Record step-by-step history for specific episodes
                if ep == 1:     # Record first episode
                    self.history_ep1.append(self.V.copy())
                elif ep == 2:   # Record second episode
                    self.history_ep2.append(self.V.copy())
                elif ep == 3:   # Record third episode
                    self.history_ep3.append(self.V.copy())
                if is_first_success:    # Record first success episode
                    self.history_first_success.append(self.V.copy())
            
            # Calculate RMSE if true_v is provided
            if true_v is not None:
                rmse = (sum((self.V[s] - true_v[s])**2 for s in true_v) / len(true_v)) ** 0.5
                self.errors.append(rmse)
                
        if verbose:
            elapsed_time = time.time() - start_time
            print(f"Monte Carlo prediction completed {num_episodes} episodes in {elapsed_time:.4f} seconds.")
            print(f"Goal reached {goals_reached} times ({(goals_reached / num_episodes) * 100:.2f}%).")
            
        return self.V
