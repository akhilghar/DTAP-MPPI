# mppi_baseline.py

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from dataclasses import dataclass
from typing import Callable

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
        self.rollout_kernel = make_rollout_kernel_coalesced(
            dynamics_func, config.state_dim, config.control_dim
        )

        self.u_nominal = np.zeros((config.horizon, config.control_dim), dtype=np.float32)
        self._last_expected_traj = None

        self._allocate_gpu_memory()

    def _allocate_gpu_memory(self):
        num_samples = self.config.num_samples
        horizon = self.config.horizon
        control_dim = self.config.control_dim
        state_dim = self.config.state_dim

        self.d_samples = cuda.device_array((horizon, num_samples, control_dim), dtype=np.float32)
        self.d_trajectories = cuda.device_array((num_samples, horizon + 1, state_dim), dtype=np.float32)
        self.d_costs = cuda.device_array(num_samples, dtype=np.float32)
        self.d_weights = cuda.device_array(num_samples, dtype=np.float32)
        self.d_weights_norm = cuda.device_array(num_samples, dtype=np.float32)
        self.d_min_dists = cuda.device_array(num_samples, dtype=np.float32)
        self.d_expected_traj = cuda.device_array((horizon + 1, state_dim), dtype=np.float32)
        self.d_expected_controls = cuda.device_array((horizon, control_dim), dtype=np.float32)

        self.d_Q_diag = cuda.to_device(np.diag(self.config.Q).astype(np.float32))
        self.d_R_diag = cuda.to_device(np.diag(self.config.R).astype(np.float32))
        self.d_Qf_diag = cuda.to_device(np.diag(self.config.Qf).astype(np.float32))
        self.d_params = cuda.to_device(self.config.dynamics_params.astype(np.float32))
        self.d_u_min = cuda.to_device(self.config.u_min.astype(np.float32))
        self.d_u_max = cuda.to_device(self.config.u_max.astype(np.float32))
        self.d_noise_sigma = cuda.to_device(self.config.noise_sigma.astype(np.float32))

        self.d_x0 = cuda.device_array(state_dim, dtype=np.float32)
        self.d_x_goal = cuda.device_array(state_dim, dtype=np.float32)
        self.d_u_nominal = cuda.device_array((horizon, control_dim), dtype=np.float32)

        seed = int(np.random.randint(1, 2**31))
        self._sample_rng_states = create_xoroshiro128p_states(num_samples, seed=seed)

        self.d_dummy_vec2 = cuda.to_device(np.zeros((1, 2), dtype=np.float32))
        self.d_dummy_float1 = cuda.to_device(np.zeros(1, dtype=np.float32))
        self.d_dummy_int1 = cuda.to_device(np.zeros(1, dtype=np.int32))

        self.d_obs_circles_positions = None
        self.d_obs_circles_radii = None
        self.d_obs_rects_positions = None
        self.d_obs_rects_width = None
        self.d_obs_rects_height = None
        self.d_obs_rects_angles = None
        self.d_obs_polys_vertices = None
        self.d_obs_polys_starts = None
        self.d_obs_polys_lengths = None

        total_bytes = (
            self.d_samples.nbytes + self.d_trajectories.nbytes +
            self.d_costs.nbytes + self.d_weights.nbytes +
            self.d_min_dists.nbytes + self.d_expected_traj.nbytes +
            self.d_expected_controls.nbytes
        )
        print(f"Allocated GPU memory: {total_bytes / 1e6:.2f} MB")

    def _get_obstacle_data(self):
        obstacles = self.environment.get_obstacle_data()
        return obstacles

    def _sync_device_array(self, attr_name: str, host_array: np.ndarray):
        host_array = np.ascontiguousarray(host_array)
        dev_array = getattr(self, attr_name, None)

        if (
            dev_array is None
            or tuple(dev_array.shape) != tuple(host_array.shape)
            or dev_array.dtype != host_array.dtype
        ):
            if dev_array is not None and hasattr(dev_array, 'close'):
                try:
                    dev_array.close()
                except Exception:
                    pass
            dev_array = cuda.to_device(host_array)
            setattr(self, attr_name, dev_array)
        else:
            cuda.to_device(host_array, to=dev_array)

        return dev_array

    def _sync_obstacle_buffers(self, obstacles):
        circles = obstacles['circles']
        rects = obstacles['rectangles']
        polys = obstacles['polygons']

        circle_count = int(circles['count'])
        rect_count = int(rects['count'])
        poly_count = int(polys['count'])

        if circle_count > 0:
            d_obs_circles_positions = self._sync_device_array(
                'd_obs_circles_positions', circles['positions'].astype(np.float32)
            )
            d_obs_circles_radii = self._sync_device_array(
                'd_obs_circles_radii', circles['radii'].astype(np.float32)
            )
        else:
            d_obs_circles_positions = self.d_dummy_vec2
            d_obs_circles_radii = self.d_dummy_float1

        if rect_count > 0:
            d_obs_rects_positions = self._sync_device_array(
                'd_obs_rects_positions', rects['positions'].astype(np.float32)
            )
            d_obs_rects_width = self._sync_device_array(
                'd_obs_rects_width', rects['widths'].astype(np.float32)
            )
            d_obs_rects_height = self._sync_device_array(
                'd_obs_rects_height', rects['heights'].astype(np.float32)
            )
            d_obs_rects_angles = self._sync_device_array(
                'd_obs_rects_angles', rects['angles'].astype(np.float32)
            )
        else:
            d_obs_rects_positions = self.d_dummy_vec2
            d_obs_rects_width = self.d_dummy_float1
            d_obs_rects_height = self.d_dummy_float1
            d_obs_rects_angles = self.d_dummy_float1

        if poly_count > 0:
            d_obs_polys_vertices = self._sync_device_array(
                'd_obs_polys_vertices', polys['vertices_flat'].astype(np.float32)
            )
            d_obs_polys_starts = self._sync_device_array(
                'd_obs_polys_starts', polys['starts'].astype(np.int32)
            )
            d_obs_polys_lengths = self._sync_device_array(
                'd_obs_polys_lengths', polys['lengths'].astype(np.int32)
            )
        else:
            d_obs_polys_vertices = self.d_dummy_vec2
            d_obs_polys_starts = self.d_dummy_int1
            d_obs_polys_lengths = self.d_dummy_int1

        return {
            'circle_positions': d_obs_circles_positions,
            'circle_radii': d_obs_circles_radii,
            'circle_count': circle_count,
            'rect_positions': d_obs_rects_positions,
            'rect_widths': d_obs_rects_width,
            'rect_heights': d_obs_rects_height,
            'rect_angles': d_obs_rects_angles,
            'rect_count': rect_count,
            'poly_vertices': d_obs_polys_vertices,
            'poly_starts': d_obs_polys_starts,
            'poly_lengths': d_obs_polys_lengths,
            'poly_count': poly_count,
        }
    
    # Warm-starts 
    def _rollout_nominal(self, x0: np.ndarray) -> np.ndarray:
        traj = np.zeros((self.config.horizon + 1, self.config.state_dim))
        traj[0] = x0
        return traj
    
    def solve(self, x0: np.ndarray, x_goal: np.ndarray, return_trajectory: bool = False, require_safe: bool = False):
        cuda.to_device(x0.astype(np.float32), to=self.d_x0)
        cuda.to_device(x_goal.astype(np.float32), to=self.d_x_goal)
        cuda.to_device(self.u_nominal.astype(np.float32), to=self.d_u_nominal)

        threads_per_block = 256
        blocks_per_grid = (self.config.num_samples + threads_per_block - 1) // threads_per_block

        generate_samples_kernel[blocks_per_grid, threads_per_block](
            self._sample_rng_states,
            self.d_samples,
            self.d_u_nominal,
            self.d_noise_sigma,
            self.d_u_min,
            self.d_u_max,
            self.config.num_samples,
            self.config.horizon,
            self.config.control_dim,
        )

        obstacles = self._get_obstacle_data()
        obstacle_buffers = self._sync_obstacle_buffers(obstacles)
        robot_radius = self.environment.robot_radius

        self.rollout_kernel[blocks_per_grid, threads_per_block](
            self.d_samples, self.d_trajectories, self.d_x0, self.d_params,
            self.config.dt, self.config.num_samples, self.config.horizon
        )

        static_cost_min_dist_kernel[blocks_per_grid, threads_per_block](
            self.d_trajectories, self.d_samples, self.d_costs, self.d_min_dists,
            self.d_x_goal, self.d_Q_diag, self.d_R_diag, self.d_Qf_diag,
            obstacle_buffers['circle_positions'], obstacle_buffers['circle_radii'], obstacle_buffers['circle_count'],
            obstacle_buffers['rect_positions'], obstacle_buffers['rect_widths'], obstacle_buffers['rect_heights'], obstacle_buffers['rect_angles'], obstacle_buffers['rect_count'],
            obstacle_buffers['poly_vertices'], obstacle_buffers['poly_starts'], obstacle_buffers['poly_lengths'], obstacle_buffers['poly_count'],
            self.config.d_safe, self.config.Q_obs, robot_radius,
            self.config.num_samples, self.config.horizon, self.config.state_dim, self.config.control_dim, self.environment.bounds
        )

        costs = self.d_costs.copy_to_host()

        is_safe = True

        if require_safe:
            min_dists = self.d_min_dists.copy_to_host()
            safe_mask = min_dists >= self.config.d_safe

            if not np.any(safe_mask):
                best_idx = int(np.argmax(min_dists))
                samples_host = self.d_samples.copy_to_host()
                u_nominal_best = samples_host[:, best_idx, :].astype(np.float32)
                u_opt = u_nominal_best[0, :]
                self.u_nominal = u_nominal_best
                traj_best = self.d_trajectories[best_idx].copy_to_host() if return_trajectory else None
                self._last_expected_traj = traj_best
                is_safe = False
                if return_trajectory:
                    return u_opt, traj_best, is_safe
                return u_opt, None, is_safe

            else:
                costs[~safe_mask] += 1e6

        min_cost = np.min(costs)
        weights = np.exp(-(costs - min_cost) / max(1e-8, self.config.lambda_)).astype(np.float32)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights_sum

        cuda.to_device(weights, to=self.d_weights)

        total_ctrl_elems = self.config.horizon * self.config.control_dim
        expected_controls_coalesced_kernel[total_ctrl_elems, threads_per_block](
            self.d_samples, self.d_weights, self.d_expected_controls,
            self.config.num_samples, self.config.horizon, self.config.control_dim
        )

        trajectory = None
        if return_trajectory:
            total_traj_elems = (self.config.horizon + 1) * self.config.state_dim
            expected_trajectory_kernel[total_traj_elems, threads_per_block](
                self.d_trajectories, self.d_weights, self.d_expected_traj,
                self.config.num_samples, self.config.horizon, self.config.state_dim
            )
            trajectory = self.d_expected_traj.copy_to_host()
            self._last_expected_traj = trajectory
        else:
            self._last_expected_traj = None

        expected_controls = self.d_expected_controls.copy_to_host()
        self.u_nominal = expected_controls
        u_opt = expected_controls[0, :].astype(np.float32)

        if return_trajectory:
            return u_opt, trajectory, is_safe
        return u_opt, None, is_safe
        
    def get_control(self, x0: np.ndarray, x_goal: np.ndarray, require_safe: bool = True):
        u_opt, _, is_safe = self.solve(x0, x_goal, return_trajectory=False, require_safe=require_safe)
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
            'd_min_dists', 'd_expected_traj', 'd_expected_controls',
            'd_Q_diag', 'd_R_diag', 'd_Qf_diag', 'd_params',
            'd_u_min', 'd_u_max', 'd_noise_sigma',
            'd_x0', 'd_x_goal', 'd_u_nominal', '_sample_rng_states',
            'd_dummy_vec2', 'd_dummy_float1', 'd_dummy_int1',
            'd_obs_circles_positions', 'd_obs_circles_radii',
            'd_obs_rects_positions', 'd_obs_rects_width', 'd_obs_rects_height', 'd_obs_rects_angles',
            'd_obs_polys_vertices', 'd_obs_polys_starts', 'd_obs_polys_lengths'
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