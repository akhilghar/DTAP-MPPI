# mppi_baseline.py

import numpy as np
from numba import cuda, jit
import math
from dataclasses import dataclass
from typing import Callable, Optional

from controllers.cuda_kernels import *

@dataclass
class MPPIConfig:
    # Dimensions
    state_dim: int = 4
    control_dim: int = 2

    # MPPI Parameters
    num_samples: int = 1000
    horizon: int = 30
    dt: float = 0.1

    # Cost Parameters
    Q: np.ndarray = None  # State cost matrix
    R: np.ndarray = None  # Control cost matrix
    Qf: np.ndarray = None  # Terminal state cost matrix
    lambda_: float = 1.0  # Temperature parameter for MPPI

    # Obstacle parameters
    Q_obs: float = 100.0  # Obstacle cost weight
    d_safe: float = 0.5  # Safe distance from obstacles
    obs_cost_type: str = 'barrier'  # Type of obstacle cost function

    # Dynamics parameters
    dynamics_params: np.ndarray = None

    # Saturation limits
    u_min: np.ndarray = None
    u_max: np.ndarray = None

    # Sampling parameters
    noise_sigma: np.ndarray = None

    def __post_init__(self):
        if self.Q is None:
            self.Q = np.eye(self.state_dim)
        if self.R is None:
            self.R = 0.01*np.eye(self.control_dim)
        if self.Qf is None:
            self.Qf = 10*np.eye(self.state_dim)
        if self.u_min is None:
            self.u_min = -np.ones(self.control_dim)
        if self.u_max is None:
            self.u_max = np.ones(self.control_dim)
        if self.noise_sigma is None:
            self.noise_sigma = 0.5 * np.ones(self.control_dim)
        if self.dynamics_params is None:
            self.dynamics_params = np.array([1.0], dtype=np.float32)  # Default parameters for dynamics (e.g., wheelbase)


class MPPIBaseline:
    def __init__(self, config: MPPIConfig, dynamics_func: Callable, environment):

        if not hasattr(dynamics_func, 'metadata'):
            raise ValueError("Dynamics function must have metadata for state_dim, control_dim, and params_dim.")
        
        metadata = dynamics_func.metadata
        
        print("Automatically detected dynamics function metadata:")
        print(f"  Name: {metadata['name']}")
        print(f"  Description: {metadata['description']}")
        print(f"  State Dimension: {metadata['state_dim']}")
        print(f"  Control Dimension: {metadata['control_dim']}")
        print(f"  Parameters Dimension: {metadata['params_dim']}")

        if config.dynamics_params is not None and len(config.dynamics_params) != metadata['params_dim']:
            raise ValueError(f"Provided dynamics_params has dimension {len(config.dynamics_params)}, but expected {metadata['params_dim']} based on the dynamics function metadata.")

        self.config = config
        self.environment = environment

        self.config.state_dim = metadata['state_dim']
        self.config.control_dim = metadata['control_dim']

        if not cuda.is_available():
            print("CUDA is not available. MPPI will run on CPU, which may be slow.")

        print(f"CUDA Available: {cuda.is_available()}")
        print(f"CUDA devices: {cuda.gpus}")

        print(f"Compiling Rollout Kernel for dynamics: {metadata['name']}...")
        self.rollout_kernel = make_rollout_kernel(dynamics_func, config.state_dim, config.control_dim)

        self.u_nominal = np.zeros((config.horizon, config.control_dim), dtype=np.float32)

        self._allocate_gpu_memory()

    def _allocate_gpu_memory(self):
        num_samples = self.config.num_samples
        horizon = self.config.horizon
        control_dim = self.config.control_dim
        state_dim = self.config.state_dim

        self.d_samples = cuda.device_array((num_samples, horizon, control_dim), dtype=np.float32)
        self.d_trajectories = cuda.device_array((num_samples, horizon + 1, state_dim), dtype=np.float32)
        self.d_costs = cuda.device_array(num_samples, dtype=np.float32)
        self.d_weights = cuda.device_array(num_samples, dtype=np.float32)

        traj_bytes = self.d_trajectories.nbytes
        cost_bytes = self.d_costs.nbytes
        samples_bytes = self.d_samples.nbytes
        weights_bytes = self.d_weights.nbytes
        total_bytes = traj_bytes + cost_bytes + samples_bytes + weights_bytes
        print(f"Allocated GPU memory: {total_bytes / 1e6:.2f} MB")
        
    def _generate_samples(self):
        noise = np.random.normal(0, self.config.noise_sigma, size=(self.config.num_samples, self.config.horizon, self.config.control_dim)).astype(np.float32)
        samples = self.u_nominal + noise
        samples = np.clip(samples, self.config.u_min, self.config.u_max)
        return samples
    
    def _get_obstacle_data(self):
        obstacles = self.environment.get_obstacle_data()
        return obstacles
    
    def _rollout_nominal(self, x0: np.ndarray) -> np.ndarray:
        traj = np.zeros((self.config.horizon + 1, self.config.state_dim))
        traj[0] = x0
        return traj
    
    def solve(self, x0: np.ndarray, x_goal: np.ndarray, return_trajectory: bool = False):
        samples = self._generate_samples()
        
        d_samples = cuda.to_device(samples)
        d_x0 = cuda.to_device(x0.astype(np.float32))
        d_x_goal = cuda.to_device(x_goal.astype(np.float32))
        d_params = cuda.to_device(self.config.dynamics_params.astype(np.float32))

        d_Q_diag = cuda.to_device(np.diag(self.config.Q).astype(np.float32))
        d_R_diag = cuda.to_device(np.diag(self.config.R).astype(np.float32))
        d_Qf_diag = cuda.to_device(np.diag(self.config.Qf).astype(np.float32))

        obstacles = self._get_obstacle_data()

        # Circles
        d_obs_circles_positions = cuda.to_device(np.array(obstacles['circles']['positions'], dtype=np.float32))
        d_obs_circles_radii = cuda.to_device(np.array(obstacles['circles']['radii'], dtype=np.float32))
        d_obs_circles_count = int(obstacles['circles']['count'])

        # Rectangles
        d_obs_rects_positions = cuda.to_device(np.array(obstacles['rectangles']['positions'], dtype=np.float32))
        d_obs_rects_width = cuda.to_device(np.array(obstacles['rectangles']['widths'], dtype=np.float32))
        d_obs_rects_height = cuda.to_device(np.array(obstacles['rectangles']['heights'], dtype=np.float32))
        d_obs_rects_angles = cuda.to_device(np.array(obstacles['rectangles']['angles'], dtype=np.float32))
        d_obs_rects_count = int(obstacles['rectangles']['count'])

        # Polygons: use flattened vertices, starts and lengths
        # 'vertices_flat' is (N_total, 2), 'starts' is int32 array, 'lengths' is int32 array
        d_obs_polys_vertices = cuda.to_device(np.array(obstacles['polygons']['vertices_flat'], dtype=np.float32))
        d_obs_polys_starts = cuda.to_device(np.array(obstacles['polygons']['starts'], dtype=np.int32))
        d_obs_polys_lengths = cuda.to_device(np.array(obstacles['polygons']['lengths'], dtype=np.int32))
        d_obs_polys_count = int(obstacles['polygons']['count'])

        robot_radius = self.environment.robot_radius

        threads_per_block = 256
        blocks_per_grid = (self.config.num_samples + threads_per_block - 1) // threads_per_block

        self.rollout_kernel[blocks_per_grid, threads_per_block](
            d_samples, self.d_trajectories, d_x0, d_params,
            self.config.dt, self.config.num_samples, self.config.horizon
        )

        cost_kernel[blocks_per_grid, threads_per_block](
            self.d_trajectories, d_samples, self.d_costs, d_x_goal, d_Q_diag, d_R_diag, d_Qf_diag,
            d_obs_circles_positions, d_obs_circles_radii, d_obs_circles_count,
            d_obs_rects_positions, d_obs_rects_width, d_obs_rects_height, d_obs_rects_angles, d_obs_rects_count, 
            d_obs_polys_vertices, d_obs_polys_starts, d_obs_polys_lengths, d_obs_polys_count,
            robot_radius, self.config.Q_obs, self.config.d_safe,
            self.config.num_samples, self.config.horizon, self.config.state_dim, self.config.control_dim
        )

        costs = self.d_costs.copy_to_host()
        compute_weights_kernel[blocks_per_grid, threads_per_block](
            self.d_costs, self.d_weights, self.config.lambda_, self.config.num_samples
        )
        weights = self.d_weights.copy_to_host()
        sum_weights = np.sum(weights) + 1e-6  # Avoid division by zero
        weights /= sum_weights
        u_opt = np.sum(samples[:, 0, :] * weights[:, np.newaxis], axis=0) / (np.sum(weights))

        weighted_samples = samples * weights[:, np.newaxis, np.newaxis]
        u_nominal_new = np.sum(weighted_samples, axis=0)
        self.u_nominal = u_nominal_new

        if return_trajectory:
            trajectory = self._rollout_nominal(x0)
            return u_opt, trajectory
        else:
            return u_opt, None
        
    def get_control(self, x0: np.ndarray, x_goal: np.ndarray, require_safe: bool = True):
        u_opt, trajectory = self.solve(x0, x_goal, return_trajectory=True)
        is_safe = True
        if require_safe and trajectory is not None:
            for t in range(trajectory.shape[0]):
                pos = trajectory[t, :2]
                dist = self.environment.get_nearest_obstacle_distance(pos)
                if dist < self.config.d_safe:
                    is_safe = False
                    break
        return u_opt, is_safe