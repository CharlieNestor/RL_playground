import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from envs.maze_env import MazeEnv

def plot_grid(
    env: MazeEnv, 
    algorithm: str,
    V: Dict[Tuple[int, int], float], 
    iteration: int) -> None:
    """
    Renders the maze environment grid and overlays the current state values V(s).
    Colors the values green if they exceed half the goal reward, and red if they
    drop below half the minimum negative value on the board.
    
    :param env: The maze environment object containing layout and reward information.
    :param algorithm: The name of thealgorithm used to evaluate the policy.
    :param V: A dictionary mapping states (row, col) to their current estimated value.
    :param iteration: The current iteration number of the policy evaluation.
    """
    # Make figsize proportional to the grid dimensions
    fig, ax = plt.subplots(figsize=(env.width, env.height))
    
    # Set grid lines exactly on the edges of the cells (-0.5, 0.5, 1.5, etc.)
    ax.set_xticks(np.arange(-0.5, env.width, 1))
    ax.set_yticks(np.arange(-0.5, env.height, 1))
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.invert_yaxis()
    
    # Hide the tick labels but keep the grid lines
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', length=0)
    ax.grid(color='black', linestyle='-', linewidth=2)
    ax.set_title(f"Iterative {algorithm} - Iteration {iteration}", fontsize=14)
    
    # Dynamic thresholds based on environment and current values
    threshold_green = env.reward_goal / 2.0
    
    negative_values = [v for v in V.values() if v < 0.0]
    if negative_values:
        min_negative = min(negative_values)
        threshold_red = min_negative / 2.0
    else:
        threshold_red = -1e-5  # Default threshold so exactly 0.0 isn't red
    
    # Draw the maze elements and values
    for r in range(env.height):
        for c in range(env.width):
            val = env.maze[r][c]
            
            if val == 'X':
                # Draw a filled dark block for walls that scales automatically
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#404040'))
            elif val == 'G':
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#ccffcc')) # Light green highlight for Goal
            elif val == 'S':
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#ffffcc')) # Light yellow for Start
            
            if val not in ['X', 'G']:
                v_val = V.get((r, c), 0.0)
                
                # Dynamic coloring for the text to indicate goodness/badness
                color = 'black'
                if v_val > threshold_green: 
                    color = 'green'
                elif v_val < threshold_red: 
                    color = 'red'
                    
                ax.text(c, r, f"{v_val:.2f}", ha='center', va='center', 
                        fontsize=12, fontweight='bold', color=color)
                
    plt.show()


def plot_policy(
    env: MazeEnv, 
    algorithm: str,
    policy: Dict[Tuple[int, int], Dict[int, float]]) -> None:
    """
    Renders the maze environment grid and overlays the optimal policy using arrows.
    
    :param env: The maze environment object containing layout and reward information.
    :param algorithm: The name of the algorithm used to find the policy.
    :param policy: A dictionary mapping states (row, col) to their action probabilities.
    """
    # Make figsize proportional to the grid dimensions
    fig, ax = plt.subplots(figsize=(env.width, env.height))
    
    # Set grid lines exactly on the edges of the cells (-0.5, 0.5, 1.5, etc.)
    ax.set_xticks(np.arange(-0.5, env.width, 1))
    ax.set_yticks(np.arange(-0.5, env.height, 1))
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.invert_yaxis()
    
    # Hide the tick labels but keep the grid lines
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', length=0)
    ax.grid(color='black', linestyle='-', linewidth=2)
    ax.set_title(f"Optimal Policy ({algorithm})", fontsize=14)
    
    # Action to Arrow Mapping
    # UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3
    action_to_arrow = {
        0: '↑',
        1: '↓',
        2: '←',
        3: '→'
    }
    
    # Draw the maze elements and values
    for r in range(env.height):
        for c in range(env.width):
            val = env.maze[r][c]
            
            if val == 'X':
                # Draw a filled dark block for walls that scales automatically
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#404040'))
            elif val == 'G':
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#ccffcc')) # Light green highlight for Goal
            elif val == 'S':
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#ffffcc')) # Light yellow for Start
            
            if val not in ['X', 'G']:
                action_probs = policy.get((r, c))
                if action_probs is not None:
                    # Find action with highest probability
                    best_action = max(action_probs, key=action_probs.get)
                    arrow = action_to_arrow.get(best_action, '')
                    ax.text(c, r, arrow, ha='center', va='center', 
                            fontsize=20, fontweight='bold', color='blue')
                
    plt.show()
