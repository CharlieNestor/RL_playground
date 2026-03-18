import numpy as np
import pandas as pd
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
    policy: Dict[Tuple[int, int], Dict[int, float]],
    wind_map: dict = None) -> None:
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
            elif wind_map and (r, c) in wind_map:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#e6f2ff')) # Light blue for Wind
            
            if val not in ['X', 'G']:
                action_probs = policy.get((r, c))
                if action_probs is not None:
                    # Find action with highest probability
                    best_action = max(action_probs, key=action_probs.get)
                    arrow = action_to_arrow.get(best_action, '')
                    ax.text(c, r, arrow, ha='center', va='center', 
                            fontsize=20, fontweight='bold', color='blue')
                
    plt.show()

def plot_combined_chart(
    env: MazeEnv, 
    algorithm: str, 
    path: list, 
    episode: int, 
    rewards_up_to_now: list,
    lengths_up_to_now: list = None,
    wind_map: dict = None) -> None:
    """
    Renders both the maze path and the learning curve side-by-side.
    
    :param env: The maze environment object.
    :param algorithm: The name of the algorithm.
    :param path: A list of state tuples (row, col) visited during the episode.
    :param episode: The current episode number.
    :param rewards_up_to_now: The rewards up to current episode.
    :param lengths_up_to_now: The step counts up to current episode.
    """
    if lengths_up_to_now:
        rewards_plot = [r / l if l > 0 else 0 for r, l in zip(rewards_up_to_now, lengths_up_to_now)]
    else:
        rewards_plot = rewards_up_to_now

    lengths_plot = lengths_up_to_now

    # Calculate proportional widths (e.g. 0.8 inch per grid cell for maze, 8 inches for chart)
    maze_width = max(env.width * 0.8, 4)
    chart_width = 8
    total_width = maze_width + chart_width
    
    # Set up a 1x2 subplot grid with proportional width ratios
    fig, (ax1, ax2) = plt.subplots(
        1, 2, 
        figsize=(total_width, max(env.height * 0.8, 5)),
        gridspec_kw={'width_ratios': [maze_width, chart_width]}
    )
    
    # --- AXIS 1: PATH Visualization ---
    ax1.set_xticks(np.arange(-0.5, env.width, 1))
    ax1.set_yticks(np.arange(-0.5, env.height, 1))
    ax1.set_xlim(-0.5, env.width - 0.5)
    ax1.set_ylim(-0.5, env.height - 0.5)
    ax1.invert_yaxis()
    
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.tick_params(axis='both', length=0)
    ax1.grid(color='black', linestyle='-', linewidth=2)
    ax1.set_title(f"{algorithm} - Path (Episode {episode})", fontsize=14)
    
    # Draw maze layout
    for r in range(env.height):
        for c in range(env.width):
            val = env.maze[r][c]
            if val == 'X':
                ax1.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#404040'))
            elif val == 'G':
                ax1.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#ccffcc'))
            elif val == 'S':
                ax1.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#ffffcc'))
            elif wind_map and (r, c) in wind_map:
                ax1.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#e6f2ff'))

    # Draw path
    if path:
        cols = [p[1] for p in path]
        rows = [p[0] for p in path]
        ax1.plot(cols, rows, color='blue', linewidth=3, alpha=0.6, marker='o', markersize=5)
        ax1.plot(cols[-1], rows[-1], color='red', marker='o', markersize=5) 
        
    # --- AXIS 2: REWARDS Visualization ---
    episodes = np.arange(1, len(rewards_plot) + 1)
    ax2.plot(episodes, rewards_plot, color='green', linewidth=2, label='Reward')
    
    ax2.set_title(f"Performance per Episode (Up to Ep {episode})", fontsize=14)
    ax2.set_xlabel("Episode")
    
    # Cap the x-axis to the next 100 to avoid constant resizing
    x_max = max(100, ((episode - 1) // 100 + 1) * 100)
    ax2.set_xlim(1, x_max)
    
    # Handle Y-axis scale and label
    label_base = "Avg Reward/Step" if lengths_up_to_now else "Total Reward"
    ax2.set_ylabel(label_base, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Handle Secondary Y-axis for Episode Lengths
    if lengths_plot is not None:
        ax3 = ax2.twinx()
        ax3.plot(episodes, lengths_plot, color='red', linewidth=2, alpha=0.7, label='Steps')
        ax3.set_ylabel("Step Count", color='red')
        ax3.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    plt.show()
