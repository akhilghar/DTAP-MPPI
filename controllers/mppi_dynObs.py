# mppi_dynObs.py

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from dataclasses import dataclass
from typing import Callable, Optional
from scipy.signal import savgol_filter

from controllers.cuda_kernels import (
    make_rollout_kernel_coalesced,
    generate_samples_kernel,
    mc_cost_and_min_dist_kernel,
    obs_mc_rollout_kernel,
    expected_trajectory_kernel,
    expected_controls_coalesced_kernel,
    sample_terrain_kernel
)


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
    d_safe: float = 0.5   # Safe distance from obstacles
    obs_cost_type: str = 'barrier'

    # Terrain parameters
    Q_slope: float = 15.0  # Cost weight for terrain slope (if used in dynamics)
    Q_elev: float = 4.0   # Cost weight for elevation (if used in dynamics)

    # Terrain sensing parameters
    sensor_radius: float = 1.0
    sensor_offset: float = 1.5
    n_terrain_points: int = 100  # Number of points to sample for terrain estimation

    # Dynamics parameters
    dynamics_params: np.ndarray = None

    # Saturation limits
    u_min: np.ndarray = None
    u_max: np.ndarray = None

    # Sampling parameters
    noise_sigma: np.ndarray = None

    # Covariance adaptation parameters
    covariance_max_scale: float = 4.0   # maximum multiplier on base sigma when in danger
    covariance_decay: float = 0.7       # per-step decay rate back toward base when recovering

    # Probabilistic obstacle prediction parameters
    obs_pred_rollouts: int = 40          # MC rollouts for GPU obstacle trajectory prediction
    obs_direction_change_prob: float = 0.05
    max_obstacles: int = 20              # capacity for preallocated obstacle GPU buffers

    def __post_init__(self):
        if self.Q is None:
            self.Q = np.eye(self.state_dim)
        if self.R is None:
            self.R = 0.01 * np.eye(self.control_dim)
        if self.Qf is None:
            self.Qf = 10 * np.eye(self.state_dim)
        if self.u_min is None:
            self.u_min = -np.ones(self.control_dim)
        if self.u_max is None:
            self.u_max = np.ones(self.control_dim)
        if self.noise_sigma is None:
            self.noise_sigma = 0.5 * np.ones(self.control_dim)
        if self.dynamics_params is None:
            self.dynamics_params = np.array([1.0], dtype=np.float32)


class MPPIDynObs:
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
            raise ValueError(
                f"Provided dynamics_params has dimension {len(config.dynamics_params)}, "
                f"but expected {metadata['params_dim']} based on the dynamics function metadata."
            )

        self.config = config
        self.environment = environment
        xmin, _, ymin, _ = self.environment.bounds
        self.terrain_info = np.array([xmin, ymin, self.environment.dx, self.environment.dy], dtype=np.float32)

        self.config.state_dim   = metadata['state_dim']
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

        self.base_covariance    = self.config.noise_sigma.copy()
        self.current_covariance = self.config.noise_sigma.copy()

        # RNG states for obs_mc_rollout_kernel — lazily grown as obstacle count changes
        self._obs_rng_states   = None
        self._obs_rng_n_states = 0

        self._allocate_gpu_memory()

    # ------------------------------------------------------------------
    # GPU memory management
    # ------------------------------------------------------------------

    def _allocate_gpu_memory(self):
        N   = self.config.num_samples
        H   = self.config.horizon
        C   = self.config.control_dim
        S   = self.config.state_dim
        R   = self.config.obs_pred_rollouts
        N_c = self.config.max_obstacles

        # ---- Main compute buffers ----
        # samples layout transposed to (H, N, C) for coalesced warp access
        self.d_samples      = cuda.device_array((H, N, C),     dtype=np.float32)
        self.d_trajectories = cuda.device_array((N, H + 1, S), dtype=np.float32)
        self.d_costs        = cuda.device_array(N,              dtype=np.float32)
        self.d_min_dists    = cuda.device_array(N,              dtype=np.float32)
        self.d_weights      = cuda.device_array(N,              dtype=np.float32)
        self.d_weights_norm = cuda.device_array(N,              dtype=np.float32)
        self.d_expected_traj     = cuda.device_array((H + 1, S), dtype=np.float32)
        self.d_expected_controls = cuda.device_array((H, C),      dtype=np.float32)

        # ---- Fixed-content buffers (uploaded once at init, never change) ----
        self.d_Q_diag  = cuda.to_device(np.diag(self.config.Q).astype(np.float32))
        self.d_R_diag  = cuda.to_device(np.diag(self.config.R).astype(np.float32))
        self.d_Qf_diag = cuda.to_device(np.diag(self.config.Qf).astype(np.float32))
        self.d_params  = cuda.to_device(self.config.dynamics_params.astype(np.float32))
        self.d_u_min   = cuda.to_device(self.config.u_min.astype(np.float32))
        self.d_u_max   = cuda.to_device(self.config.u_max.astype(np.float32))

        # ---- Per-call small buffers (updated each solve via to= parameter) ----
        self.d_x0       = cuda.device_array(S, dtype=np.float32)
        self.d_x_goal   = cuda.device_array(S, dtype=np.float32)
        self.d_sigma    = cuda.device_array(C, dtype=np.float32)
        self.d_u_nominal = cuda.device_array((H, C), dtype=np.float32)

        # ---- Sample generation RNG (one state per sample thread) ----
        seed = int(np.random.randint(1, 2**31))
        self._sample_rng_states = create_xoroshiro128p_states(N, seed=seed)

        # ---- Obstacle state buffers (at max capacity; first N_obs slots used) ----
        self.d_obs_pos   = cuda.device_array((N_c, 2), dtype=np.float32)
        self.d_obs_vel   = cuda.device_array((N_c, 2), dtype=np.float32)
        self.d_obs_spd   = cuda.device_array(N_c,      dtype=np.float32)
        self.d_obs_radii = cuda.device_array(N_c,      dtype=np.float32)

        # Host-side staging arrays for obstacle uploads (avoids numpy alloc per call)
        self._h_obs_pos   = np.zeros((N_c, 2), dtype=np.float32)
        self._h_obs_vel   = np.zeros((N_c, 2), dtype=np.float32)
        self._h_obs_spd   = np.zeros(N_c,      dtype=np.float32)
        self._h_obs_radii = np.zeros(N_c,      dtype=np.float32)

        # ---- MC obstacle rollout buffer: (R, max_obs, H, 2) ----
        self.d_obs_rollouts = cuda.device_array((R, N_c, H, 2), dtype=np.float32)

        # ---- Static dummy buffers for empty rect/poly args ----
        self.d_dummy_vec2   = cuda.to_device(np.zeros((1, 2), dtype=np.float32))
        self.d_dummy_float1 = cuda.to_device(np.zeros(1,      dtype=np.float32))
        self.d_dummy_int1   = cuda.to_device(np.zeros(1,      dtype=np.int32))

        # ---- Terrain buffers ----
        self.d_terrain = cuda.to_device(self.environment.terrain.astype(np.float32))
        self.d_terrain_info = cuda.to_device(self.terrain_info)

        self.d_terrain_xy = cuda.device_array((self.config.n_terrain_points, 2), dtype=np.float32)
        self.d_terrain_elev = cuda.device_array(self.config.n_terrain_points, dtype=np.float32)
        terrain_seed = int(np.random.randint(1, 2**31))
        self._terrain_rng_states = create_xoroshiro128p_states(self.config.n_terrain_points, seed=terrain_seed)

        self._last_expected_traj = None

        total_bytes = (
            self.d_samples.nbytes + self.d_trajectories.nbytes +
            self.d_costs.nbytes   + self.d_min_dists.nbytes    +
            self.d_weights.nbytes + self.d_obs_rollouts.nbytes +
            self.d_terrain.nbytes + self.d_terrain_info.nbytes
        )
        print(f"Allocated GPU memory: {total_bytes / 1e6:.2f} MB")

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def solve(self, x0: np.ndarray, x_goal: np.ndarray,
              return_trajectory: bool = False, require_safe: bool = False):

        # ---- Upload per-call scalars/vectors (tiny H2D, no allocation) ----
        cuda.to_device(x0.astype(np.float32),                  to=self.d_x0)
        cuda.to_device(x_goal.astype(np.float32),              to=self.d_x_goal)
        cuda.to_device(self.current_covariance.astype(np.float32), to=self.d_sigma)
        cuda.to_device(self.u_nominal.astype(np.float32),      to=self.d_u_nominal)

        threads = 256
        blocks  = (self.config.num_samples + threads - 1) // threads

        # ---- Generate samples on GPU (eliminates CPU RNG + 8 MB H2D) ----
        generate_samples_kernel[blocks, threads](
            self._sample_rng_states, self.d_samples, self.d_u_nominal,
            self.d_sigma, self.d_u_min, self.d_u_max,
            self.config.num_samples, self.config.horizon, self.config.control_dim,
        )

        # ---- Obstacle MC rollouts ----
        obstacles     = self.environment.get_obstacle_data()
        circles       = obstacles['circles']
        circle_count  = circles['count']

        if circle_count > 0:
            if circle_count > self.config.max_obstacles:
                raise ValueError(
                    f"Obstacle count {circle_count} exceeds max_obstacles="
                    f"{self.config.max_obstacles}. Increase MPPIConfig.max_obstacles."
                )

            R = self.config.obs_pred_rollouts
            N = circle_count

            # Populate host-side staging buffers (no per-call numpy alloc)
            for i, obs in enumerate(self.environment.obstacles):
                self._h_obs_pos[i]   = obs.position
                self._h_obs_vel[i]   = obs.velocity
                self._h_obs_spd[i]   = float(np.linalg.norm(obs.velocity))
                self._h_obs_radii[i] = obs.radius

            cuda.to_device(self._h_obs_pos,   to=self.d_obs_pos)
            cuda.to_device(self._h_obs_vel,   to=self.d_obs_vel)
            cuda.to_device(self._h_obs_spd,   to=self.d_obs_spd)
            cuda.to_device(self._h_obs_radii, to=self.d_obs_radii)

            # Lazily (re)allocate obs RNG states if R×N has grown
            n_rng = R * N
            if self._obs_rng_states is None or n_rng > self._obs_rng_n_states:
                seed = int(np.random.randint(1, 2**31))
                self._obs_rng_states   = create_xoroshiro128p_states(n_rng, seed=seed)
                self._obs_rng_n_states = n_rng

            xmin, xmax, ymin, ymax = self.environment.bounds
            blocks_mc = (n_rng + threads - 1) // threads
            obs_mc_rollout_kernel[blocks_mc, threads](
                self._obs_rng_states,
                self.d_obs_pos, self.d_obs_vel, self.d_obs_spd, self.d_obs_radii,
                np.float32(xmin), np.float32(xmax),
                np.float32(ymin), np.float32(ymax),
                self.d_obs_rollouts, R, N, self.config.horizon,
                np.float32(self.config.dt),
                np.float32(self.config.obs_direction_change_prob),
            )

            d_circle_pred  = self.d_obs_rollouts
            d_circle_radii = self.d_obs_radii
        else:
            R = 0
            d_circle_pred  = cuda.to_device(np.zeros((1, 1, 1, 2), dtype=np.float32))
            d_circle_radii = cuda.to_device(np.zeros(1,             dtype=np.float32))

        # ---- Rectangles ----
        rects      = obstacles['rectangles']
        rect_count = rects['count']
        if rect_count > 0:
            d_rect_pos    = cuda.to_device(rects['positions'])
            d_rect_widths = cuda.to_device(rects['widths'])
            d_rect_heights= cuda.to_device(rects['heights'])
            d_rect_angles = cuda.to_device(rects['angles'])
        else:
            d_rect_pos    = self.d_dummy_vec2
            d_rect_widths = self.d_dummy_float1
            d_rect_heights= self.d_dummy_float1
            d_rect_angles = self.d_dummy_float1

        # ---- Polygons ----
        polys      = obstacles['polygons']
        poly_count = polys['count']
        if poly_count > 0:
            d_poly_verts   = cuda.to_device(polys['vertices_flat'])
            d_poly_starts  = cuda.to_device(polys['starts'])
            d_poly_lengths = cuda.to_device(polys['lengths'])
        else:
            d_poly_verts   = self.d_dummy_vec2
            d_poly_starts  = self.d_dummy_int1
            d_poly_lengths = self.d_dummy_int1

        robot_radius = self.environment.robot_radius

        # ---- Trajectory rollout ----
        self.rollout_kernel[blocks, threads](
            self.d_samples, self.d_trajectories, self.d_x0, self.d_params,
            self.config.dt, self.d_terrain, self.d_terrain_info, self.config.num_samples, self.config.horizon,
        )

        # ---- Terrain sampling ----
        robot_x = x0[0]
        robot_y = x0[1]
        robot_theta = x0[2]
        if np.abs(robot_theta) > np.pi:
            if robot_theta > 0:
                robot_theta -= np.pi
            else:                
                robot_theta += np.pi

        robot_z = self.get_robot_elevation(robot_x, robot_y)

        sensed_cx = robot_x + self.config.sensor_offset * np.cos(robot_theta)
        sensed_cy = robot_y + self.config.sensor_offset * np.sin(robot_theta)

        N_t = self.config.n_terrain_points
        sample_terrain_kernel[1, N_t](
            self._terrain_rng_states, self.d_terrain, self.d_terrain_xy, self.d_terrain_elev,
            float(robot_x), float(robot_y), float(robot_theta), float(self.config.sensor_offset),
            float(self.config.sensor_radius), int(N_t), float(self.environment.bounds[0]),
            float(self.environment.bounds[2]), float(self.environment.dx)
        )

        # ---- Single-pass cost + min_dist ----
        mc_cost_and_min_dist_kernel[blocks, threads](
            self.d_trajectories, self.d_samples, self.d_costs, self.d_min_dists,
            self.d_x_goal, self.d_Q_diag, self.d_R_diag, self.d_Qf_diag,
            d_circle_pred, d_circle_radii, int(circle_count), R,
            d_rect_pos, d_rect_widths, d_rect_heights, d_rect_angles, int(rect_count),
            d_poly_verts, d_poly_starts, d_poly_lengths, int(poly_count),
            self.config.d_safe, self.config.Q_obs, robot_radius,
            self.config.num_samples, self.config.horizon,
            self.config.state_dim, self.config.control_dim,
            self.environment.bounds, 
            self.d_terrain_xy, self.d_terrain_elev, int(N_t),
            float(sensed_cx), float(sensed_cy), float(self.config.sensor_radius),
            float(self.config.Q_slope), float(self.config.Q_elev), float(robot_z)
        )

        # ---- D2H: costs and min_dists (triggers GPU sync) ----
        costs      = self.d_costs.copy_to_host()
        min_dists  = self.d_min_dists.copy_to_host()

        is_safe = True

        if require_safe:
            safe_mask     = min_dists >= self.config.d_safe
            safe_fraction = float(np.mean(safe_mask))
            danger        = 1.0 - safe_fraction

            if safe_fraction == 0.0:
                # Fully trapped: maximise exploration and return best-clearance control.
                self.current_covariance = self.base_covariance * self.config.covariance_max_scale

                best_idx = int(np.argmax(min_dists))
                # Full samples D2H only in this rare escape path (~8 MB, one-time)
                samples_host = self.d_samples.copy_to_host()       # (H, N, C)
                u_nominal_best  = samples_host[:, best_idx, :]     # (H, C)
                u_opt           = u_nominal_best[0, :].astype(np.float32)
                self.u_nominal  = u_nominal_best

                traj_best = self.d_trajectories[best_idx].copy_to_host()
                self._last_expected_traj = traj_best
                is_safe = False
                if return_trajectory:
                    return u_opt, traj_best, is_safe
                return u_opt, None, is_safe

            # Penalise unsafe samples
            costs[~safe_mask] += 1e6

            if danger > 0.0:
                target = self.base_covariance * (
                    1.0 + (self.config.covariance_max_scale - 1.0) * danger
                )
                self.current_covariance = np.maximum(self.current_covariance, target)
                self.config.covariance_decay = 0.3 + 0.6 * danger**3
            else:
                self.current_covariance = (
                    self.config.covariance_decay * self.current_covariance
                    + (1.0 - self.config.covariance_decay) * self.base_covariance
                )

        # ---- CPU softmin weights ----
        min_cost   = np.min(costs)
        weights    = np.exp(-(costs - min_cost) / max(1e-8, self.config.lambda_)).astype(np.float32)
        w_sum      = np.sum(weights)
        weights    = weights / w_sum if w_sum > 0 else np.ones_like(weights) / len(weights)

        # ---- Expected trajectory and controls on GPU (use preallocated buffers) ----
        cuda.to_device(weights, to=self.d_weights)

        # One block per output element — threads within each block reduce over num_samples
        total_traj_elems = (self.config.horizon + 1) * self.config.state_dim
        expected_trajectory_kernel[total_traj_elems, 256](
            self.d_trajectories, self.d_weights, self.d_expected_traj,
            self.config.num_samples, self.config.horizon, self.config.state_dim,
        )

        total_ctrl_elems = self.config.horizon * self.config.control_dim
        expected_controls_coalesced_kernel[total_ctrl_elems, 256](
            self.d_samples, self.d_weights, self.d_expected_controls,
            self.config.num_samples, self.config.horizon, self.config.control_dim,
        )

        self._last_expected_traj = self.d_expected_traj.copy_to_host()
        trajectory = self._last_expected_traj if return_trajectory else None
        expected_controls = self.d_expected_controls.copy_to_host()   # (H, C)
        # print(expected_controls)

        window_size = 20
        window_deg = 3
        self.u_nominal[:, 0] = savgol_filter(expected_controls[:, 0], window_size, window_deg)
        self.u_nominal[:, 1] = savgol_filter(expected_controls[:, 1], window_size, window_deg)
        u_opt = expected_controls[0, :].astype(np.float32)

        if return_trajectory:
            return u_opt, trajectory, is_safe
        return u_opt, None, is_safe
    
    def get_robot_elevation(self, px, py):
        xmin, _, ymin, _ = self.environment.bounds
        cell_size = float(self.environment.dx)

        gx = (px - xmin) / cell_size
        gy = (py - ymin) / cell_size

        ix = int(np.floor(gx))
        iy = int(np.floor(gy))
        ix = np.clip(ix, 0, self.environment.terrain.shape[1] - 2)
        iy = np.clip(iy, 0, self.environment.terrain.shape[0] - 2)

        fx = gx - ix
        fy = gy - iy

        elev = (
            self.environment.terrain[iy, ix] * (1 - fx) * (1 - fy) +
            self.environment.terrain[iy, ix + 1] * fx * (1 - fy) +
            self.environment.terrain[iy + 1, ix] * (1 - fx) * fy +
            self.environment.terrain[iy + 1, ix + 1] * fx * fy
        )
        return elev

    def get_control(self, x0: np.ndarray, x_goal: np.ndarray, require_safe: bool = True):
        u_opt, trajectory, is_safe = self.solve(
            x0, x_goal, return_trajectory=True, require_safe=require_safe
        )

        # Double-check expected trajectory clearance if available
        if require_safe and trajectory is not None:
            for t in range(trajectory.shape[0]):
                pos = trajectory[t, :2]
                dist = self.environment.get_nearest_obstacle_distance(pos)
                if dist < self.config.d_safe:
                    is_safe = False
                    break

        return u_opt, is_safe

    def get_rollout_snapshot(self, n: int = 5):
        """Return (expected_traj, sample_trajs) from the most recent solve.

        expected_traj : (H+1, S) weighted-mean (selected) trajectory.
        sample_trajs  : (n, H+1, S) array of n random sample rollouts.
        """
        indices = np.random.choice(self.config.num_samples, size=n, replace=False)
        sample_trajs = np.stack([self.d_trajectories[i].copy_to_host() for i in indices])
        return self._last_expected_traj, sample_trajs
    
    def get_covariance(self):
        return self.current_covariance

    def free_gpu_buffers(self, force_reset: bool = False):
        import gc
        from numba import cuda as _cuda

        names = (
            'd_samples', 'd_trajectories', 'd_costs', 'd_min_dists',
            'd_weights', 'd_weights_norm', 'd_expected_traj', 'd_expected_controls',
            'd_Q_diag', 'd_R_diag', 'd_Qf_diag', 'd_params',
            'd_u_min', 'd_u_max', 'd_x0', 'd_x_goal', 'd_sigma', 'd_u_nominal',
            'd_obs_pos', 'd_obs_vel', 'd_obs_spd', 'd_obs_radii',
            'd_obs_rollouts', 'd_dummy_vec2', 'd_dummy_float1', 'd_dummy_int1',
        )

        for n in names:
            if hasattr(self, n):
                try:
                    arr = getattr(self, n)
                    if arr is not None and hasattr(arr, 'close'):
                        try:
                            arr.close()
                        except Exception:
                            pass
                    setattr(self, n, None)
                except Exception:
                    setattr(self, n, None)

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
