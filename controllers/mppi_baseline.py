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
    Q_obs: float = 100.0 # Obstacle cost weight
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
            self.noise_sigma = 1.0 * np.ones(self.control_dim)
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
        # print("MPPI Configuration:", self.config)
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
        # device weight buffers (raw and normalized)
        self.d_weights = cuda.device_array(num_samples, dtype=np.float32)
        self.d_weights_norm = cuda.device_array(num_samples, dtype=np.float32)

        # preallocate buffers used by safety and expected-value kernels
        self.d_min_dists = cuda.device_array(num_samples, dtype=np.float32)
        self.d_expected_traj = cuda.device_array((horizon + 1, state_dim), dtype=np.float32)
        self.d_expected_controls = cuda.device_array((horizon, control_dim), dtype=np.float32)

        traj_bytes = self.d_trajectories.nbytes
        cost_bytes = self.d_costs.nbytes
        samples_bytes = self.d_samples.nbytes
        weights_bytes = self.d_weights.nbytes
        total_bytes = traj_bytes + cost_bytes + samples_bytes + weights_bytes
        print(f"Allocated GPU memory: {total_bytes / 1e6:.2f} MB")
    
    # Sample control perturbations and add to nominal control sequence
    def _generate_samples(self):
        noise = np.random.normal(0, self.config.noise_sigma, size=(self.config.num_samples, self.config.horizon, self.config.control_dim)).astype(np.float32)
        samples = self.u_nominal + noise
        samples = np.clip(samples, self.config.u_min, self.config.u_max)
        return samples
    
    # Helper function to retrieve obstacle data from the environment and prepare it for GPU use
    def _get_obstacle_data(self):
        obstacles = self.environment.get_obstacle_data()
        return obstacles
    
    # Warm-starts 
    def _rollout_nominal(self, x0: np.ndarray) -> np.ndarray:
        traj = np.zeros((self.config.horizon + 1, self.config.state_dim))
        traj[0] = x0
        return traj
    
    def solve(self, x0: np.ndarray, x_goal: np.ndarray, return_trajectory: bool = False, require_safe: bool = False):
        samples = self._generate_samples() # Generate samples on CPU (small)
        
        d_samples = cuda.to_device(samples)
        d_x0 = cuda.to_device(x0.astype(np.float32))
        d_x_goal = cuda.to_device(x_goal.astype(np.float32))
        d_params = cuda.to_device(self.config.dynamics_params.astype(np.float32))

        d_Q_diag = cuda.to_device(np.diag(self.config.Q).astype(np.float32))
        d_R_diag = cuda.to_device(np.diag(self.config.R).astype(np.float32))
        d_Qf_diag = cuda.to_device(np.diag(self.config.Qf).astype(np.float32))

        obstacles = self._get_obstacle_data()

        # -------- Circles --------
        circles = obstacles['circles']
        circle_count = circles['count']

        if circle_count > 0:
            d_obs_circles_positions = cuda.to_device(circles['positions'])
            d_obs_circles_radii = cuda.to_device(circles['radii'])
        else:
            d_obs_circles_positions = cuda.to_device(np.zeros((1,2), dtype=np.float32))
            d_obs_circles_radii = cuda.to_device(np.zeros(1, dtype=np.float32))

        d_obs_circles_count = int(circle_count)

        # -------- Rectangles --------
        rects = obstacles['rectangles']
        rect_count = rects['count']

        if rect_count > 0:
            d_obs_rects_positions = cuda.to_device(rects['positions'])
            d_obs_rects_width = cuda.to_device(rects['widths'])
            d_obs_rects_height = cuda.to_device(rects['heights'])
            d_obs_rects_angles = cuda.to_device(rects['angles'])
        else:
            d_obs_rects_positions = cuda.to_device(np.zeros((1,2), dtype=np.float32))
            d_obs_rects_width = cuda.to_device(np.zeros(1, dtype=np.float32))
            d_obs_rects_height = cuda.to_device(np.zeros(1, dtype=np.float32))
            d_obs_rects_angles = cuda.to_device(np.zeros(1, dtype=np.float32))

        d_obs_rects_count = int(rect_count)

        # -------- Polygons --------
        polys = obstacles['polygons']
        poly_count = polys['count']

        if poly_count > 0:
            d_obs_polys_vertices = cuda.to_device(polys['vertices_flat'])
            d_obs_polys_starts = cuda.to_device(polys['starts'])
            d_obs_polys_lengths = cuda.to_device(polys['lengths'])
        else:
            d_obs_polys_vertices = cuda.to_device(np.zeros((1,2), dtype=np.float32))
            d_obs_polys_starts = cuda.to_device(np.zeros(1, dtype=np.int32))
            d_obs_polys_lengths = cuda.to_device(np.zeros(1, dtype=np.int32))

        d_obs_polys_count = int(poly_count)


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
            self.config.d_safe, self.config.Q_obs, robot_radius,
            self.config.num_samples, self.config.horizon, self.config.state_dim, self.config.control_dim, self.environment.bounds
        )

        # Copy costs to host (small) for CPU weighting
        costs = self.d_costs.copy_to_host()

        num_samples = self.config.num_samples
        is_safe = True

        if require_safe:
            # Compute per-sample minimum clearance on GPU and copy back just the distances
            d_min_dists = cuda.device_array(num_samples, dtype=np.float32)
            min_distance_kernel[blocks_per_grid, threads_per_block](
                self.d_trajectories, d_min_dists,
                d_obs_circles_positions, d_obs_circles_radii, d_obs_circles_count,
                d_obs_rects_positions, d_obs_rects_width, d_obs_rects_height, d_obs_rects_angles, d_obs_rects_count,
                d_obs_polys_vertices, d_obs_polys_starts, d_obs_polys_lengths, d_obs_polys_count,
                robot_radius, self.config.num_samples, self.config.horizon
            )

            min_dists = d_min_dists.copy_to_host()
            safe_mask = min_dists >= self.config.d_safe

            if not np.any(safe_mask):
                # No fully-safe samples — pick the sample with maximum clearance
                best_idx = int(np.argmax(min_dists))
                u_opt = samples[best_idx, 0, :].astype(np.float32)
                try:
                    # copy only the selected trajectory to host
                    traj_best = self.d_trajectories[best_idx].copy_to_host()
                    self.u_nominal = samples[best_idx].copy()
                except Exception:
                    traj_best = None
                is_safe = False
                if return_trajectory:
                    return u_opt, traj_best, is_safe
                else:
                    return u_opt, None, is_safe
            else:
                # Penalize unsafe samples so they receive negligible weight
                costs[~safe_mask] += 1e6

        # Compute weights on CPU (softmin)
        min_cost = np.min(costs)
        weights = np.exp(-(costs - min_cost) / max(1e-8, self.config.lambda_)).astype(np.float32)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights_sum

        u_opt = np.sum(samples[:, 0, :] * weights[:, np.newaxis], axis=0).astype(np.float32)

        # Compute expected trajectory and nominal controls on GPU using the computed weights
        d_weights_dev = cuda.to_device(weights)
        d_expected_traj = cuda.device_array((self.config.horizon + 1, self.config.state_dim), dtype=np.float32)
        total_traj_elems = (self.config.horizon + 1) * self.config.state_dim
        blocks_traj = (total_traj_elems + threads_per_block - 1) // threads_per_block
        expected_trajectory_kernel[blocks_traj, threads_per_block](
            self.d_trajectories, d_weights_dev, d_expected_traj,
            self.config.num_samples, self.config.horizon, self.config.state_dim
        )

        d_expected_controls = cuda.device_array((self.config.horizon, self.config.control_dim), dtype=np.float32)
        total_ctrl_elems = self.config.horizon * self.config.control_dim
        blocks_ctrl = (total_ctrl_elems + threads_per_block - 1) // threads_per_block
        expected_controls_kernel[blocks_ctrl, threads_per_block](
            d_samples, d_weights_dev, d_expected_controls,
            self.config.num_samples, self.config.horizon, self.config.control_dim
        )

        trajectory = d_expected_traj.copy_to_host()
        u_nominal_new = d_expected_controls.copy_to_host()
        self.u_nominal = u_nominal_new

        if return_trajectory:
            return u_opt, trajectory, is_safe
        else:
            return u_opt, None, is_safe
        
    def get_control(self, x0: np.ndarray, x_goal: np.ndarray, require_safe: bool = True):
        u_opt, trajectory, is_safe = self.solve(x0, x_goal, return_trajectory=True, require_safe=require_safe)

        # Double-check expected trajectory clearance if available
        if require_safe and trajectory is not None:
            for t in range(trajectory.shape[0]):
                pos = trajectory[t, :2]
                dist = self.environment.get_nearest_obstacle_distance(pos)
                if dist < self.config.d_safe:
                    is_safe = False
                    break

        return u_opt, is_safe

    def free_gpu_buffers(self, force_reset: bool = False):
        """Free preallocated GPU buffers held by this MPPIBaseline instance.

        Args:
            force_reset: If True, call `cuda.current_context().reset()` after freeing buffers.
                         This is disruptive — kernels and contexts will need reinitialization.
        """
        import gc
        from numba import cuda as _cuda

        names = (
            'd_samples', 'd_trajectories', 'd_costs', 'd_weights', 'd_weights_norm',
            'd_min_dists', 'd_expected_traj', 'd_expected_controls'
        )

        for n in names:
            if hasattr(self, n):
                try:
                    arr = getattr(self, n)
                    if arr is not None:
                        try:
                            # preferred: call close() if available
                            if hasattr(arr, 'close'):
                                try:
                                    arr.close()
                                except Exception:
                                    pass
                        finally:
                            try:
                                delattr(self, n)
                            except Exception:
                                try:
                                    delattr(self, n)
                                except Exception:
                                    setattr(self, n, None)
                except Exception:
                    try:
                        setattr(self, n, None)
                    except Exception:
                        pass

        # Make sure GPU operations finished and drop Python references
        try:
            _cuda.synchronize()
        except Exception:
            pass
        gc.collect()

        if force_reset:
            try:
                _cuda.current_context().reset()
            except Exception:
                pass