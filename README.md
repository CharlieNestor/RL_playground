# RL Playground

This project is a Reinforcement Learning playground dedicated to implementing classical RL algorithms in a clear and instructive way. The implementations adhere to the theoretical foundations laid out in the Sutton & Barto textbook ("Reinforcement Learning: An Introduction") and the broader history of RL.

A core focus of this repository is **visualization**, helping to visually understand how these algorithms learn and operate step-by-step.

## Implemented Algorithms

The codebase includes tabular implementations of fundamental RL algorithms, categorized as follows:

- **Dynamic Programming (Model-Based)**
  - Policy Evaluation
  - Policy Iteration
  - Value Iteration
- **Model-Free Evaluation**
  - Monte Carlo Evaluation
  - Temporal Difference (TD) Evaluation
- **Model-Free Control**
  - Monte Carlo Control
  - SARSA (On-Policy TD Control)
  - Q-Learning (Off-Policy TD Control)

## Repository Structure

- `algorithms/`: Contains the pure, clear Python implementations of the classical RL algorithms.
- `envs/`: Defines the environments used to train and test the algorithms.
- `utils/`: Contains utilities, particularly the visualization functions.
- *Jupyter Notebooks*: Several notebooks (`.ipynb`) that tie together the algorithms, environments, and visualizations to provide an interactive understanding of the learning processes.

## Getting Started

To explore the playground, ensure you have the necessary dependencies installed. You can install them via:

```bash
pip install -r requirements.txt
```

Then, open the Jupyter notebooks to see the algorithms in action with step-by-step visual representations!
