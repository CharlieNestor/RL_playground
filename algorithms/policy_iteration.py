import time
from algorithms.policy_evaluation import BaseMazeDPAlgorithm

class PolicyIteration(BaseMazeDPAlgorithm):
    """
    Solves for the optimal Policy pi*(a|s) using Policy Iteration.
    Inherits from BaseMazeDPAlgorithm to reuse environment, initialization, and evaluation logic.
    """
    
    def policy_iteration(self, initial_policy: dict = None, return_history: bool = False, verbose: bool = True):
        """
        Iteratively evaluates and improves the policy until it stabilizes.
        
        :param initial_policy: Optional starting policy (stochastic or deterministic). 
                               If None, a default deterministic policy is created.
        :param return_history: If True, returns a history list of (V, policy) pairs.
        :param verbose: If True, prints progress.
        :return: A tuple of (optimal_V, optimal_policy) or (optimal_V, optimal_policy, history) if return_history is True.
        """
        # 1. Initialize policy
        is_deterministic = False
        if initial_policy is not None:
            # Copy to avoid modifying the original dictionary passed in
            policy = {s: a_probs.copy() for s, a_probs in initial_policy.items()}
        else:
            is_deterministic = True
            policy = {}
            for r in range(self.env.height):
                for c in range(self.env.width):
                    if self.env.maze[r][c] not in ['G', 'X']:
                        # Default to always picking action 0 (UP)
                        policy[(r, c)] = {a: 1.0 if a == self.actions[0] else 0.0 for a in self.actions}

        iteration = 0
        start_time = time.time()
        history = []
        
        while True:
            iteration += 1
            if verbose:
                print(f"--- Policy Iteration: Step {iteration} ---")
            
            # 2. Policy Evaluation
            # We call the inherited evaluate_policy method to compute V for the current policy
            self.evaluate_policy(policy, return_history=False, verbose=False)
            
            # Save the current policy and its evaluated Value function
            if return_history:
                history.append((self.V.copy(), {s: a_probs.copy() for s, a_probs in policy.items()}))
            
            # 3. Policy Improvement
            policy_stable = True
            
            # We iterate through all non-terminal states and improve the policy greedily
            for r in range(self.env.height):
                for c in range(self.env.width):
                    state = (r, c)
                    
                    if self.env.maze[r][c] == 'G' or self.env.maze[r][c] == 'X':
                        continue
                        
                    # If deterministic, we can find the old best action to check for stability
                    if is_deterministic:
                        old_best_action = max(policy[state], key=policy[state].get)
                    else:
                        old_best_action = None
                        
                    # 1. Compute Q-values Q(s, a) for all possible actions in this state
                    # Q(s, a) = R(s, a) + gamma * V(s')
                    q_values = {}
                    for action in self.actions:
                        next_state, reward = self._get_transition(state, action)
                        q_values[action] = reward + self.gamma * self.V[next_state]
                        
                    # 2. Policy Improvement Step (Greedy)
                    # pi'(s) = argmax_a Q(s, a)
                    best_action = max(q_values, key=q_values.get)
                            
                    # Classical stopping condition check
                    if is_deterministic and old_best_action != best_action:
                        policy_stable = False
                        
                    # Update policy to be greedy with respect to the new value function
                    policy[state] = {a: 1.0 if a == best_action else 0.0 for a in self.actions}
            
            if not is_deterministic:
                # The policy just became deterministic, so we must evaluate it at least once more
                policy_stable = False
                is_deterministic = True
                
            if policy_stable:
                elapsed_time = time.time() - start_time
                print(f"Policy Iteration converged after {iteration} iterations. Time taken: {elapsed_time:.4f} seconds")
                break
                
        if return_history:
            return self.V, policy, history
        return self.V, policy
