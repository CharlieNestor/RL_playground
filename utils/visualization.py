import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

def plot_q_values(
    env: MazeEnv, 
    algorithm: str,
    Q: Dict[Tuple[int, int], Dict[int, float]],
    iteration: int,
    show_values: bool = True,
    show_best_action: bool = True) -> None:
    """
    Renders the maze environment grid and overlays the Q-values using four colored triangles per cell.
    
    :param env: The maze environment object.
    :param algorithm: The name of the algorithm.
    :param Q: A dictionary mapping states (row, col) to a dictionary of action values.
    :param iteration: The current iteration number (or episode).
    :param show_values: If True, prints the exact Q-value number in each triangle.
    :param show_best_action: If True, draws an arrow in the center of each cell pointing to the max Q-value direction.
    """
    fig, ax = plt.subplots(figsize=(env.width, env.height))
    
    ax.set_xticks(np.arange(-0.5, env.width, 1))
    ax.set_yticks(np.arange(-0.5, env.height, 1))
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', length=0)
    ax.grid(color='black', linestyle='-', linewidth=2)
    ax.set_title(f"{algorithm} - Q-Values (Episode {iteration})", fontsize=14)
    
    # Extract all Q-values to find min and max for normalization
    all_q = []
    for state_q in Q.values():
        all_q.extend(state_q.values())
        
    vmin = min(all_q) if all_q else -1.0
    vmax = max(all_q) if all_q else 1.0
    
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
        
    cmap = plt.cm.RdYlGn
    
    # Calculate a valid linthresh based on the smallest non-zero magnitude
    magnitudes = [abs(q) for q in all_q if abs(q) > 1e-5]
    linthresh = min(magnitudes) if magnitudes else 0.01
    
    # Logarithmic scale handles exponential decay (gamma) from highly varying rewards
    # SymLogNorm works symmetrically for both high positive rewards and deep negative penalties
    norm = mcolors.SymLogNorm(linthresh=max(linthresh, 1e-5), vmin=vmin, vmax=vmax, base=10)

    action_to_arrow = {
        0: '↑',
        1: '↓',
        2: '←',
        3: '→'
    }

    def get_triangle(r, c, action):
        if action == 0: return [(c, r), (c - 0.5, r - 0.5), (c + 0.5, r - 0.5)] # UP
        if action == 1: return [(c, r), (c + 0.5, r + 0.5), (c - 0.5, r + 0.5)] # DOWN
        if action == 2: return [(c, r), (c - 0.5, r + 0.5), (c - 0.5, r - 0.5)] # LEFT
        if action == 3: return [(c, r), (c + 0.5, r - 0.5), (c + 0.5, r + 0.5)] # RIGHT
        return []

    for r in range(env.height):
        for c in range(env.width):
            val = env.maze[r][c]
            
            if val == 'X':
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#404040'))
            elif val == 'G':
                # Light green background for goal
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#ccffcc'))
                ax.text(c, r, "G", ha='center', va='center', fontsize=20, fontweight='bold')
            elif val == 'S':
                # Light yellow background for start
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#ffffcc'))

            if val not in ['X', 'G'] and (r, c) in Q:
                state_q = Q.get((r, c), {})
                
                # 1. Draw the Colored Triangles
                for action, q_val in state_q.items():
                    triangle = get_triangle(r, c, action)
                    if triangle:
                        color = cmap(norm(q_val))
                        poly = plt.Polygon(triangle, facecolor=color, edgecolor='black', linewidth=0.5)
                        ax.add_patch(poly)
                        
                        # Only show values if flag is enabled
                        if show_values:
                            dx, dy = 0, 0
                            if action == 0: dy = -0.3
                            if action == 1: dy = 0.3
                            if action == 2: dx = -0.3
                            if action == 3: dx = 0.3
                            
                            n_val = norm(q_val)
                            text_color = 'white' if n_val < 0.2 or n_val > 0.8 else 'black'
                            
                            ax.text(c + dx, r + dy, f"{q_val:.1f}", ha='center', va='center', 
                                    fontsize=8, color=text_color, fontweight='bold')
                        
                # 2. Draw separating lines
                ax.plot([c - 0.5, c + 0.5], [r - 0.5, r + 0.5], color='black', linewidth=0.5)
                ax.plot([c - 0.5, c + 0.5], [r + 0.5, r - 0.5], color='black', linewidth=0.5)
                
                # 3. Outline the cell
                ax.plot([c - 0.5, c + 0.5, c + 0.5, c - 0.5, c - 0.5],
                        [r - 0.5, r - 0.5, r + 0.5, r + 0.5, r - 0.5],
                        color='black', linewidth=1)
                
                # 4. Draw Best Action Arrow in the center
                if show_best_action and state_q:
                    best_action = max(state_q, key=state_q.get)
                    arrow = action_to_arrow.get(best_action, '')
                    # We make the arrow bold, large, and black to stand out in the middle
                    ax.text(c, r, arrow, ha='center', va='center', fontsize=20, 
                            fontweight='bold', color='black', 
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='circle,pad=0.1'))

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Q-Value")

    plt.show()

