import random
from typing import List, Tuple, Dict, Optional
from envs.maze_env import MazeEnv, UP, DOWN, LEFT, RIGHT

class WindyMazeEnv(MazeEnv):
    """
    A 2D Grid Maze Environment with Wind.
    Wind pushes the agent in a specific direction with a certain probability
    after the agent takes an action.
    """
    def __init__(
        self, 
        maze_layout: List[str], 
        reward_step: float, 
        reward_goal: float, 
        reward_wall: float,
        wind_map: Optional[Dict[Tuple[int, int], Tuple[int, float]]] = None
    ):
        """
        :param maze_layout: List of strings representing the grid 
        :param reward_step: Reward per step
        :param reward_goal: Reward for reaching the Goal
        :param reward_wall: Penalty for hitting a Wall
        :param wind_map: Dictionary mapping state (row, col) to (wind_direction, probability)
                         e.g., {(1, 1): (RIGHT, 0.5)} means cell (1, 1) has a 50% chance of 
                         blowing the agent to the right in addition to their move.
        """
        super().__init__(maze_layout, reward_step, reward_goal, reward_wall)
        
        self.wind_map = wind_map if wind_map is not None else {}

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Executes a step in the environment by applying the agent's action and environmental wind.
        
        The final position is determined by summing two forces together:
        1. The agent's intended action.
        2. The wind's displacement (stochastic).
        
        Crucially, the wind's probability and direction are calculated based on the 
        agent's CURRENT state (where they start the step), NOT from the intermediate 
        state they would reach by simply applying the policy's action. These two vectors 
        are summed to find the final hypothetical destination before a single collision 
        check is performed against walls and boundaries.
        """
        r, c = self.current_state
        
        # 1. Apply Agent's Intended Action
        if action == UP: r -= 1
        elif action == DOWN: r += 1
        elif action == LEFT: c -= 1
        elif action == RIGHT: c += 1
            
        # 2. Apply Wind Effect based on CURRENT state (before moving)
        # This models wind pushing you as you try to make your move.
        if self.current_state in self.wind_map:
            wind_dir, wind_prob = self.wind_map[self.current_state]
            if random.random() < wind_prob:
                if wind_dir == UP: r -= 1
                elif wind_dir == DOWN: r += 1
                elif wind_dir == LEFT: c -= 1
                elif wind_dir == RIGHT: c += 1
        
        # 3. Check boundary and wall collisions (short-circuit logic)
        hit_boundary = r < 0 or r >= self.height or c < 0 or c >= self.width
        
        if hit_boundary or self.maze[r][c] == 'X':
            # Invalid move: stay in the exact same place as before the step, assign wall penalty
            next_state = self.current_state
            reward = self.reward_wall
            done = False
        else:
            # Valid move: update state
            next_state = (r, c)
            self.current_state = next_state
            
            # Check for terminal state and assign reward
            if self.maze[r][c] == 'G':
                reward = self.reward_goal
                done = True
            else:
                reward = self.reward_step
                done = False
                
        return next_state, reward, done
