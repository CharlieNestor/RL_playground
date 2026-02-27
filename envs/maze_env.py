from typing import List, Tuple

# Action Space
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_NAMES = {
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT"
}

class MazeEnv:
    """
    A 2D Grid Maze Environment.
    """
    def __init__(
        self, 
        maze_layout: List[str], 
        reward_step: float, 
        reward_goal: float, 
        reward_wall: float
    ):
        """
        :param maze_layout: List of strings representing the grid 
                            ('S': Start, 'G': Goal, 'X': Wall, '.': Path)
        :param reward_step: Reward per step
        :param reward_goal: Reward for reaching the Goal
        :param reward_wall: Penalty for hitting a Wall
        """
        self.maze = [list(row) for row in maze_layout]
        self.height = len(self.maze)
        self.width = len(self.maze[0])
        
        self.reward_step = reward_step
        self.reward_goal = reward_goal
        self.reward_wall = reward_wall
        
        self.start_state = None
        self.goal_state = None
        
        # Locate Start and Goal
        for r in range(self.height):
            for c in range(self.width):
                if self.maze[r][c] == 'S':
                    self.start_state = (r, c)
                elif self.maze[r][c] == 'G':
                    self.goal_state = (r, c)
                    
        if self.start_state is None or self.goal_state is None:
            raise ValueError("Maze must have a Start ('S') and a Goal ('G').")
            
        self.current_state = self.start_state
        
    def reset(self) -> Tuple[int, int]:
        """
        Resets environment to its initial state.
        :return state: Starting (row, col) coordinates
        """
        self.current_state = self.start_state
        return self.current_state
        
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Executes a discrete action in the environment.
        :param action: Integer representing UP, DOWN, LEFT, or RIGHT
        :return next_state: New (row, col) position
        :return reward: Action reward
        :return done: True if Goal is reached, else False
        """
        r, c = self.current_state
        
        # Determine intended next state
        if action == UP: r -= 1
        elif action == DOWN: r += 1
        elif action == LEFT: c -= 1
        elif action == RIGHT: c += 1
            
        # Check boundary and wall collisions
        hit_boundary = r < 0 or r >= self.height or c < 0 or c >= self.width
        
        if hit_boundary or self.maze[r][c] == 'X':
            # Invalid move: stay in the same place, assign wall penalty
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

    def get_actions(self) -> List[int]:
        """
        :return actions: List of valid actions
        """
        return [UP, DOWN, LEFT, RIGHT]
