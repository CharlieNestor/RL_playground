import time
from algorithms.policy_evaluation import BaseMazeDPAlgorithm

class ValueIteration(BaseMazeDPAlgorithm):
    """
    Solves for the optimal Value Function V*(s) using Value Iteration.
    Inherits from BaseMazeDPAlgorithm to reuse environment, initialization, and transition logic.
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
        start_time = time.time()
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
                    # V*(s) = max_a Q(s,a)
                    # where Q(s,a) = R(s,a) + gamma * V(s')
                    q_values = []
                    for action in self.actions:
                        next_state, reward = self._get_transition(state, action)
                        q_values.append(reward + self.gamma * self.V[next_state])
                        
                    new_V[state] = max(q_values)
                    
                    # Track the largest change to check for convergence
                    delta = max(delta, abs(old_value - new_V[state]))
                    
            self.V = new_V
            iteration += 1
            
            if return_history:
                history.append(self.V.copy())
                
            if verbose and iteration % 20 == 0:
                print(f"Iteration {iteration}: max delta = {delta:.6f}")
                
            if delta < self.theta:
                elapsed_time = time.time() - start_time
                print(f"Value iteration converged after {iteration} iterations. Time taken: {elapsed_time:.4f} seconds")
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
                    
                # 1. Compute Q-values Q(s, a) for all possible actions in this state
                # Q(s, a) = R(s, a) + gamma * V*(s')
                q_values = {}
                for action in self.actions:
                    next_state, reward = self._get_transition(state, action)
                    q_values[action] = reward + self.gamma * self.V[next_state]
                    
                # 2. Extract optimal policy (Greedy)
                # pi*(s) = argmax_a Q(s, a)
                best_action = max(q_values, key=q_values.get)
                        
                # Create a deterministic policy that takes the best action with probability 1.0
                optimal_policy[state] = {a: 1.0 if a == best_action else 0.0 for a in self.actions}
                
        return optimal_policy
