from dataclasses import dataclass
from typing import Tuple, Callable, Optional

import numpy as np
from dataclasses import dataclass

@dataclass
class MPCConfig:
    # MPC configuration parameters
    horizon: int = 20
    dt: float = 0.1
    control_dim: int = 4
    state_dim: int = 2

    # Cost matrices
    Q: np.ndarray = None  # State cost matrix
    R: np.ndarray = None  # Control cost matrix
    Qf: np.ndarray = None  # Terminal state cost matrix

    # Obstacle parameters (if needed)
    Q_obs: float = 100.0  # Obstacle cost weight
    d_safe: float = 0.5  # Safe distance from obstacles
    obs_cost_type: str = 'barrier'  # Type of obstacle cost function

    # Saturation limits
    u_min: np.ndarray = None
    u_max: np.ndarray = None

    # Numerical Constraints
    max_iterations: int = 100
    tol: float = 1e-4

    # Base Cases for Matrices/Saturation Limits
    def __post_init__(self):
        if self.Q is None:
            self.Q = np.eye(self.state_dim)
        if self.R is None:
            self.R = np.eye(self.control_dim)
        if self.Qf is None:
            self.Qf = np.eye(self.state_dim)
        if self.u_min is None:
            self.u_min = -np.ones(self.control_dim)
        if self.u_max is None:
            self.u_max = np.ones(self.control_dim)

class MPCBaseline:
    # Baseline MPC controller using a simple quadratic cost and linear dynamics
    def __init__(self, config: MPCConfig, dynamics_func: Callable, environment):
        self.config = config
        self.dynamics_func = dynamics_func
        self.environment = environment

        self.ctrlseq = np.zeros((self.config.horizon, self.config.control_dim))

    def obstacleCostEval(self, x: np.ndarray) -> float:
        dist = self.environment.get_nearest_obstacle_distance(x)

        if self.config.obs_cost_type == 'barrier':
            if dist < 0:
                return 1e6
            elif dist < self.config.d_safe:
                # Penalize being close to the obstacle: larger penalty when distance is smaller
                return self.config.Q_obs * (self.config.d_safe - dist + 1e-6)
            else:
                return 0.0
            
        elif self.config.obs_cost_type == 'quadratic':
            if dist < 0:
                return 1e6
            elif dist < self.config.d_safe:
                return self.config.Q_obs * (self.config.d_safe - dist) ** 2
            else:
                return 0.0
            
        elif self.config.obs_cost_type == 'exponential':
            if dist < 0:
                return 1e6
            elif dist < self.config.d_safe:
                return self.config.Q_obs * np.exp(-dist / self.config.d_safe)
        
        return 0.0

    def costEval(self, xseq: np.ndarray, useq: np.ndarray, x_goal: np.ndarray) -> float:
        # Evaluate the cost of a given state and control sequence
        cost = 0.0
        for t in range(self.config.horizon):
            state_cost = (xseq[t] - x_goal).T @ self.config.Q @ (xseq[t] - x_goal)
            control_cost = useq[t].T @ self.config.R @ useq[t]
            cost += state_cost + control_cost
            pos = xseq[t,:2]  # Assuming the first two dimensions are position
            cost += self.obstacleCostEval(pos)  # Add obstacle cost based on position
        terminal_cost = (xseq[-1] - x_goal).T @ self.config.Qf @ (xseq[-1] - x_goal)
        terminal_pos = xseq[-1,:2]
        cost += terminal_cost
        cost += self.obstacleCostEval(terminal_pos)  # Add obstacle cost for terminal state
        return cost
    
    def rollout(self, x0: np.ndarray, useq: np.ndarray) -> np.ndarray:
        # Rollout the system dynamics given an initial state and control sequence
        xseq = np.zeros((self.config.horizon + 1, self.config.state_dim))
        xseq[0] = x0
        for t in range(self.config.horizon):
            xseq[t + 1] = self.dynamics_func(xseq[t], useq[t], self.config.dt)
        return xseq
    
    def clip_controls(self, useq: np.ndarray) -> np.ndarray:
        # Clip control inputs to their saturation limits
        return np.clip(useq, self.config.u_min, self.config.u_max)
    
    def trajectory_is_feasible(self, xseq: np.ndarray) -> bool:
        # Check if the trajectory is feasible (e.g., no collisions)
        pos = xseq[:, :2]  # Assuming the first two dimensions are position
        if self.environment.check_trajectory_collision(pos):
            return False
        return True
    
    def solve(self, x0: np.ndarray, x_goal: np.ndarray, u_init: Optional[np.ndarray] = None, require_safe: bool = False) -> np.ndarray:
        # Solve the MPC problem using a simple iterative approach
        
        if u_init is not None:
            ctrlseq = u_init.copy()
        else:
            ctrlseq = self.ctrlseq.copy()  # Start with the current control sequence

        best_safe_cost = float('inf')
        best_safe_ctrlseq = None

        best_unsafe_cost = float('inf')
        best_unsafe_ctrlseq = ctrlseq.copy()

        for iteration in range(self.config.max_iterations):
            # Rollout the system with the current control sequence
            xseq = self.rollout(x0, ctrlseq)

            # Evaluate the cost and safety of the current trajectory
            cost = self.costEval(xseq, ctrlseq, x_goal)
            is_safe = self.trajectory_is_feasible(xseq)

            # Track best safe and best unsafe solutions separately
            if is_safe:
                if cost < best_safe_cost:
                    best_safe_cost = cost
                    best_safe_ctrlseq = ctrlseq.copy()
            else:
                if cost < best_unsafe_cost:
                    best_unsafe_cost = cost
                    best_unsafe_ctrlseq = ctrlseq.copy()

            # Use central-difference gradient approximation for better accuracy
            grad = np.zeros_like(ctrlseq)
            epsilon = 1e-3

            for t in range(self.config.horizon):
                for i in range(self.config.control_dim):
                    u_plus = ctrlseq.copy()
                    u_minus = ctrlseq.copy()
                    u_plus[t, i] += epsilon
                    u_minus[t, i] -= epsilon

                    cost_plus = self.costEval(self.rollout(x0, u_plus), u_plus, x_goal)
                    cost_minus = self.costEval(self.rollout(x0, u_minus), u_minus, x_goal)

                    grad[t, i] = (cost_plus - cost_minus) / (2 * epsilon)

            alpha = 0.1  # Larger step size for faster descent
            ctrlseq -= alpha * grad
            ctrlseq = self.clip_controls(ctrlseq)

            # Check for Convergence
            if iteration > 0 and abs(cost - best_safe_cost) < self.config.tol:
                break
            
            prev_cost = cost
        
        # Choose final control sequence: prefer safe; fallback to best unsafe if none safe found
        if require_safe and best_safe_ctrlseq is None:
            chosen_ctrlseq = best_unsafe_ctrlseq
        else:
            chosen_ctrlseq = best_safe_ctrlseq if best_safe_ctrlseq is not None else best_unsafe_ctrlseq

        if chosen_ctrlseq is None:
            chosen_ctrlseq = ctrlseq.copy()

        # Shift stored ctrlseq for warm start
        self.ctrlseq[:-1] = chosen_ctrlseq[1:]
        self.ctrlseq[-1] = chosen_ctrlseq[-1]

        x_opt = self.rollout(x0, chosen_ctrlseq)
        is_safe = self.trajectory_is_feasible(x_opt)
        return chosen_ctrlseq, x_opt, is_safe
    
    def get_control(self, x0: np.ndarray, x_goal: np.ndarray, require_safe: bool = False) -> Tuple[np.ndarray, bool]:
        # Get the optimal control input for the current state and goal
        ctrlseq, xseq, is_safe = self.solve(x0, x_goal, require_safe=require_safe)
        return ctrlseq[0], is_safe  # Return the first control input in the sequence


        