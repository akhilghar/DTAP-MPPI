import numpy as np
from numba import cuda, jit
import sys

class MPPIBaseline:
    def __init__(self, horizon: int, dt: float, num_samples: int):
        self.horizon = horizon
        self.dt = dt
        self.num_samples = num_samples

    def solve(self, x0: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """
        Solve the MPPI problem given the initial state and reference trajectory.

        Parameters:
        x0 (np.ndarray): Initial state of the system.
        x_ref (np.ndarray): Reference trajectory for the horizon.

        Returns:
        np.ndarray: Optimal control inputs for the horizon.
        """
        # Sample control sequences
        control_sequences = np.random.uniform(-1, 1, (self.num_samples, self.horizon, 2))  # Assuming 2 control inputs

        # Evaluate cost for each control sequence
        costs = np.zeros(self.num_samples)
        for i in range(self.num_samples):
            costs[i] = self.evaluate_cost(x0, control_sequences[i], x_ref)

        # Compute weights based on costs
        weights = np.exp(-costs / np.min(costs))  # Softmax-like weighting

        # Compute optimal control as weighted average
        optimal_control = np.sum(control_sequences * weights[:, None, None], axis=0) / np.sum(weights)

        return optimal_control

    def evaluate_cost(self, x0: np.ndarray, control_sequence: np.ndarray, x_ref: np.ndarray) -> float:
        """
        Evaluate the cost of a given control sequence.

        Parameters:
        x0 (np.ndarray): Initial state of the system.
        control_sequence (np.ndarray): Control sequence to evaluate.
        x_ref (np.ndarray): Reference trajectory for the horizon.

        Returns:
        float: Cost of the control sequence.
        """
        cost = 0.0
        state = x0.copy()
        
        for t in range(self.horizon):
            # Simple linear dynamics for demonstration
            state += control_sequence[t] * self.dt
            cost += np.sum((state - x_ref[t]) ** 2)  # Quadratic cost to reference

        return cost