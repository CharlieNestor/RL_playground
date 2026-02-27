from envs.maze_env import MazeEnv, UP, DOWN, LEFT, RIGHT

class PolicyEvaluator:
    """
    Evaluates a given policy for a MazeEnv using Iterative Policy Evaluation.
    This calculates the Value Function V(s) for all states under a specific policy.
    """
    def __init__(self, env: MazeEnv, gamma: float = 0.99, theta: float = 1e-5):
        """
        :param env: The maze environment (assumes fully observable transition model)
        :param gamma: Discount factor (how much future rewards are valued)
        :param theta: A small threshold for determining convergence
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        # Initialize Value table: V(s) = 0 for all states
        self.V = {
            (r, c): 0.0 
            for r in range(self.env.height) 
            for c in range(self.env.width)
        }
        
        # We need to know the basic actions
        self.actions = self.env.get_actions()

    def _get_transition(self, state: tuple, action: int) -> tuple:
        """
        A helper method to simulate the environment's dynamics P(s', r | s, a)
        without actually stepping the environment state.
        ASSUMPTION: The maze is deterministic, taking an action from 'state' leads to exactly
        one 'next_state' with probability 1.0, and yields 'reward'.
        
        :param state: (row, col)
        :param action: Integer representing UP, DOWN, LEFT, or RIGHT
        :return: (next_state, reward)
        """
        r, c = state
        
        if action == UP: r -= 1
        elif action == DOWN: r += 1
        elif action == LEFT: c -= 1
        elif action == RIGHT: c += 1
            
        # Hit boundary or wall
        if r < 0 or r >= self.env.height or c < 0 or c >= self.env.width or self.env.maze[r][c] == 'X':
            return state, self.env.reward_wall
            
        next_state = (r, c)
        
        # Check if we hit the goal
        if self.env.maze[r][c] == 'G':
            return next_state, self.env.reward_goal
            
        return next_state, self.env.reward_step

    def evaluate_policy(self, policy: dict, return_history: bool = False, verbose: bool = True):
        """
        Iteratively evaluates a policy until the change in value (delta) is less than theta.
        We are doing synchronous updates to the value table, that is we first compute the new values
        for all states and then update the value table.
        
        :param policy: A dictionary mapping states to action probabilities, 
                       e.g., {(row, col): {action_int: probability, ...}}
                        If a state is not in the policy (like a wall), it's ignored.
        :param return_history: If True, returns a tuple (final_V, history_of_Vs).
        :param verbose: If True, prints the maximum change in value at each iteration.
        :return: The converged Value table V(s), or (V, history) if return_history is True
        """
        iteration = 0
        history = []
        if return_history:
            history.append(self.V.copy())
        
        while True:
            delta = 0.0
            
            # Synchronous update: First compute new values for all states, then update the value table.
            new_V = self.V.copy()
            
            for r in range(self.env.height):
                for c in range(self.env.width):
                    state = (r, c)
                    
                    # We don't evaluate terminal (Goal) states or Walls (unreachable)
                    if self.env.maze[r][c] == 'G' or self.env.maze[r][c] == 'X':
                        continue
                        
                    old_value = self.V[state]
                    
                    # Look up the action probabilities for this state
                    # e.g., {UP: 0.25, DOWN: 0.25, LEFT: 0.25, RIGHT: 0.25}
                    if state not in policy:
                        continue 
                        
                    action_probs = policy[state]
                    
                    # BELLMAN EXPECTATION EQUATION (Stochastic Policy)
                    # V(s) = sum_a pi(a|s) * [ R(s,a) + gamma * V(s') ]
                    expected_value = 0.0
                    for action, prob in action_probs.items():
                        if prob > 0:
                            next_state, reward = self._get_transition(state, action)
                            expected_value += prob * (reward + self.gamma * self.V[next_state])
                            
                    new_V[state] = expected_value
                    
                    # Track the largest change to check for convergence
                    delta = max(delta, abs(old_value - new_V[state]))
                    
            self.V = new_V
            iteration += 1
            # If return_history is True, we keep track of the value function at each iteration
            if return_history:
                history.append(self.V.copy())
            
            # Print occasional improvements
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: max delta = {delta:.6f}")
                
            if delta < self.theta:
                print(f"Policy evaluation converged after {iteration} iterations.")
                break
                
        if return_history:
            return self.V, history
        return self.V
