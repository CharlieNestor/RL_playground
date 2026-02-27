from algorithms.policy_evaluation import PolicyEvaluator

class ValueIteration(PolicyEvaluator):
    """
    Solves for the optimal Value Function V*(s) using Value Iteration.
    Inherits from PolicyEvaluator to reuse environment, initialization, and transition logic.
    """
    
    def value_iteration(self, return_history: bool = False, verbose: bool = True):
        """
        Iteratively updates the value function using the Bellman Optimality Equation
        until the change in value (delta) is less than theta.
        
        :param return_history: If True, returns a tuple (final_V, history_of_Vs).
        :param verbose: If True, prints the maximum change in value at each iteration.
        :return: The converged optimal Value table V*(s), or (V*, history) if return_history is True
        """
        iteration = 0
        history = []
        if return_history:
            history.append(self.V.copy())
            
        while True:
            delta = 0.0
            
            # Synchronous update: compute new values for all states, then update the value table.
            new_V = self.V.copy()
            
            for r in range(self.env.height):
                for c in range(self.env.width):
                    state = (r, c)
                    
                    # We don't update terminal (Goal) states or Walls (unreachable)
                    if self.env.maze[r][c] == 'G' or self.env.maze[r][c] == 'X':
                        continue
                        
                    old_value = self.V[state]
                    
                    # BELLMAN OPTIMALITY EQUATION
                    # V*(s) = max_a [ R(s,a) + gamma * V*(s') ]
                    max_expected_value = float('-inf')
                    
                    for action in self.actions:
                        next_state, reward = self._get_transition(state, action)
                        # The environment dynamics wrapper we inherit already handles deterministic transitions
                        expected_value = reward + self.gamma * self.V[next_state]
                        
                        if expected_value > max_expected_value:
                            max_expected_value = expected_value
                            
                    new_V[state] = max_expected_value
                    
                    # Track the largest change to check for convergence
                    delta = max(delta, abs(old_value - new_V[state]))
                    
            self.V = new_V
            iteration += 1
            
            if return_history:
                history.append(self.V.copy())
                
            if verbose and iteration % 20 == 0:
                print(f"Iteration {iteration}: max delta = {delta:.6f}")
                
            if delta < self.theta:
                print(f"Value iteration converged after {iteration} iterations.")
                break
                
        if return_history:
            return self.V, history
        return self.V

    def extract_optimal_policy(self):
        """
        Once the optimal value function V*(s) is computed, we can extract the deterministic 
        optimal policy pi*(a|s) by acting greedily with respect to V*(s).
        
        :return: A policy dictionary mapping states to action probabilities (1.0 for optimal action, 0.0 for others).
        """
        optimal_policy = {}
        
        for r in range(self.env.height):
            for c in range(self.env.width):
                state = (r, c)
                
                if self.env.maze[r][c] == 'G' or self.env.maze[r][c] == 'X':
                    continue
                    
                best_action = None
                max_expected_value = float('-inf')
                
                for action in self.actions:
                    next_state, reward = self._get_transition(state, action)
                    expected_value = reward + self.gamma * self.V[next_state]
                    
                    if expected_value > max_expected_value:
                        max_expected_value = expected_value
                        best_action = action
                        
                # Create a deterministic policy that takes the best action with probability 1.0
                optimal_policy[state] = {a: 1.0 if a == best_action else 0.0 for a in self.actions}
                
        return optimal_policy
