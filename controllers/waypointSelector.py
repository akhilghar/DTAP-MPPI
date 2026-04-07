import numpy as np
from typing import Tuple, Optional

class WaypointSelector:
    def __init__(self, grid_resolution: float, grid_half_size: int,
                 goal_weight: float = 1.0, obstacle_weight: float = 5.0,
                 terrain_weight: float = 3.0, heading_weight: float = 2.0,
                 d_safe: float = 0.5):
        
        self.grid_resolution = grid_resolution
        self.grid_half_size = grid_half_size
        self.goal_weight = goal_weight
        self.obstacle_weight = obstacle_weight
        self.terrain_weight = terrain_weight
        self.heading_weight = heading_weight
        self.d_safe = d_safe

        self.grid_dim = 2 * grid_half_size + 1

        offsets = np.arange(-grid_half_size, grid_half_size + 1) * grid_resolution
        ox, oy = np.meshgrid(offsets, offsets)

        self.offsets = np.stack([ox.ravel(), oy.ravel()], axis=-1).astype(np.float32)  # (N, 2)

        center_idx = grid_half_size * self.grid_dim + grid_half_size
        self.offsets = np.delete(self.offsets, center_idx, axis=0)  # Remove the center point (0,0) from the offsets

        self.current_waypoint = None  # Forces replan on first call

    def compute_heuristic(self, candidate_points: np.ndarray, robot_pos: np.ndarray, robot_heading: float,
                          goal_pos: np.ndarray, obs_positions: np.ndarray, obs_radii: np.ndarray,
                          terrain_costs: np.ndarray) -> np.ndarray:
        n = len(candidate_points)

        # Progress heuristic
        robot_goal_dist = np.linalg.norm(robot_pos - goal_pos)
        candidate_goal_dist = np.linalg.norm(candidate_points - goal_pos, axis=1)
        progress = robot_goal_dist - candidate_goal_dist  # Positive if candidate is closer to goal
        h_goal = -1*self.goal_weight * progress  # Closer to goal is better

        # Obstacle heuristic
        if len(obs_positions) > 0:
            diff = candidate_points[:, np.newaxis, :] - obs_positions[np.newaxis, :, :]  # (N, M, 2)
            dists = np.linalg.norm(diff, axis=2)  # (N, M)
            surface_dists = dists - obs_radii[np.newaxis, :]  # (N, M)
            min_obs_dist = np.min(surface_dists, axis=1)  # (N,)

            h_obs = np.where(min_obs_dist <= 0,
                             np.inf,
                             np.where(min_obs_dist < self.d_safe,
                                      self.obstacle_weight * (self.d_safe / min_obs_dist - 1.0),
                                      self.obstacle_weight * 0.1 * np.exp(-(min_obs_dist - self.d_safe) / self.d_safe)))
        else:
            h_obs = np.zeros(n, dtype=np.float32)

        # Terrain heuristic
        h_terrain = self.terrain_weight * terrain_costs

        # Heading heuristic
        dx = candidate_points[:, 0] - robot_pos[0]
        dy = candidate_points[:, 1] - robot_pos[1]
        candidate_headings = np.arctan2(dy, dx)
        angle_diff = np.abs(candidate_headings - robot_heading)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        h_heading = self.heading_weight * angle_diff

        return h_goal + h_obs + h_terrain + h_heading

    def select_waypoint(self, robot_pos: np.ndarray, robot_heading: float,
                        goal_pos: np.ndarray, obs_positions: np.ndarray,
                        obs_radii: np.ndarray, terrain_cost_fn=None) -> np.ndarray:
        
        candidate_points = (robot_pos + self.offsets)  # (N, 2)
        n = len(candidate_points)

        if len(obs_positions) > 0:
            diff = candidate_points[:, np.newaxis, :] - obs_positions[np.newaxis, :, :]  # (N, M, 2)
            dists = np.linalg.norm(diff, axis=2)  # (N, M)
            surface_dists = dists - obs_radii[np.newaxis, :]  # (N, M)
            min_obs_dist = np.min(surface_dists, axis=1)  # (N,)
            valid_mask = min_obs_dist > self.d_safe
            candidate_points = candidate_points[valid_mask]
            
            if len(candidate_points) == 0:
                # If all candidates are too close to obstacles, repel from the nearest obstacle
                nearest_idx = np.argmin(np.linalg.norm(robot_pos - obs_positions, axis=1) - obs_radii)
                away_dir = robot_pos - obs_positions[nearest_idx]
                away_dir /= (np.linalg.norm(away_dir) + 1e-6)
                self.current_waypoint = robot_pos + away_dir * self.grid_resolution
                return self.current_waypoint

        if terrain_cost_fn is not None:
            terrain_costs = terrain_cost_fn(candidate_points)  # (N,)
        else:
            terrain_costs = np.zeros(n, dtype=np.float32)

        h = self.compute_heuristic(candidate_points, robot_pos, robot_heading,
                                   goal_pos, obs_positions, obs_radii, terrain_costs)
        
        best_idx = np.argmin(h)
        self.current_waypoint = candidate_points[best_idx]

        print(f"Selected Waypoint: {self.current_waypoint}, Heuristic Value: {h[best_idx]:.4f}")

        return self.current_waypoint

    def plan_step(self, robot_pos: np.ndarray, robot_heading: float,
                  goal_pos: np.ndarray, obs_positions: np.ndarray,
                  obs_radii: np.ndarray, terrain_cost_fn=None,
                  replan_threshold: float=None) -> np.ndarray:
        
        if replan_threshold is None:
            replan_threshold = self.grid_resolution

        waypoint_blocked = False
        if self.current_waypoint is not None and len(obs_positions) > 0:
            dists_to_wp = np.linalg.norm(self.current_waypoint - obs_positions, axis=1) - obs_radii
            waypoint_blocked = np.any(dists_to_wp < self.d_safe)

        needs_replan = (
            self.current_waypoint is None or
            np.linalg.norm(self.current_waypoint - robot_pos) < replan_threshold or
            waypoint_blocked
        )

        if needs_replan:
            self.select_waypoint(robot_pos, robot_heading, goal_pos,
                                 obs_positions, obs_radii, terrain_cost_fn)
            
        return self.current_waypoint
