import numpy as np
from typing import Tuple, Callable

class Camera:
    def __init__(self, focal_length: float, sensor_size: Tuple[float, float], image_size: Tuple[int, int],
                 mounting_height: float, mounting_angle: float, baseline: float, max_range: float):
        self.focal_length = focal_length
        self.sensor_size = sensor_size
        self.image_size = image_size
        self.mounting_height = mounting_height
        self.mounting_angle = mounting_angle
        self.baseline = baseline
        self.max_range = max_range

        self.pixel_size = (sensor_size[0] / image_size[0], sensor_size[1] / image_size[1])
        self.horizontal_fov = 2 * np.arctan((sensor_size[0] / 2) / focal_length)
        self.vertical_fov = 2 * np.arctan((sensor_size[1] / 2) / focal_length)

    def compute_sensor_frustum(self, robot_position: np.ndarray, robot_heading: float) -> dict:
        blind_zone = self.mounting_height * np.tan(np.radians(self.mounting_angle) + np.radians(self.vertical_fov / 2))
        far_dist = min(self.max_range, self.mounting_height / np.cos(np.radians(self.mounting_angle) + np.radians(self.vertical_fov / 2)))
        near_dist = max(0, blind_zone)
        near_left = robot_position + np.array([near_dist * np.cos(robot_heading + np.radians(self.horizontal_fov / 2)),
                                               near_dist * np.sin(robot_heading + np.radians(self.horizontal_fov / 2))])
        near_right = robot_position + np.array([near_dist * np.cos(robot_heading - np.radians(self.horizontal_fov / 2)),
                                                near_dist * np.sin(robot_heading - np.radians(self.horizontal_fov / 2))])
        far_left = robot_position + np.array([far_dist * np.cos(robot_heading + np.radians(self.horizontal_fov / 2)),
                                              far_dist * np.sin(robot_heading + np.radians(self.horizontal_fov / 2))])
        far_right = robot_position + np.array([far_dist * np.cos(robot_heading - np.radians(self.horizontal_fov / 2)),
                                               far_dist * np.sin(robot_heading - np.radians(self.horizontal_fov / 2))])
        return {
            'near_left': near_left,
            'near_right': near_right,
            'far_left': far_left,
            'far_right': far_right,
            'near_dist': near_dist,
            'far_dist': far_dist,
            'blind_zone': blind_zone
        }
    
    def generate_pixel_grid(self, step: int = 4) -> np.ndarray:
        x_coords = np.arange(0, self.image_size[0], step)
        y_coords = np.arange(0, self.image_size[1], step)
        pixel_grid = np.stack(np.meshgrid(x_coords, y_coords), axis=-1).reshape(-1, 2)
        return pixel_grid
    
    def pixel_to_rays(self, pixel_grid: np.ndarray) -> np.ndarray:
        fx = self.focal_length / self.pixel_size[0]
        fy = self.focal_length / self.pixel_size[1]

        cx = self.image_size[0] / 2
        cy = self.image_size[1] / 2

        dx = (pixel_grid[:, 0] - cx) / fx
        dy = (pixel_grid[:, 1] - cy) / fy
        dz = np.ones_like(dx)

        dirs_cam = np.stack((dx, dy, dz), axis=-1)

        # Rotation matrix for mounting angle
        dirs_robot = np.zeros_like(dirs_cam)
        theta = np.radians(self.mounting_angle)
        
        dirs_robot[:, 0] = dirs_cam[:, 0] # x remains the same
        dirs_robot[:, 1] = dirs_cam[:, 1] * np.cos(theta) - dirs_cam[:, 2] * np.sin(theta) # y is rotated by mounting angle
        dirs_robot[:, 2] = dirs_cam[:, 1] * np.sin(theta) + dirs_cam[:, 2] * np.cos(theta) # z is rotated by mounting angle

        norms = np.linalg.norm(dirs_robot, axis=1, keepdims=True)
        dirs_robot_normalized = dirs_robot / norms

        return dirs_robot_normalized
    
    def rays_to_ground(self, rays: np.ndarray, robot_position: np.ndarray, robot_heading: float,
                       terrain_query_fn: Callable, step_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.radians(robot_heading)
        
        world_dx = rays[:, 0] * np.cos(theta) - rays[:, 2] * np.sin(theta)
        world_dy = rays[:, 0] * np.sin(theta) + rays[:, 2] * np.cos(theta)
        world_dz = rays[:, 1]

        n_rays = len(rays)
        max_steps = int(self.max_range / step_size)

        active = np.ones(n_rays, dtype=bool)
        hit_points = np.full((n_rays, 3), np.nan)
        hit_depths = np.full(n_rays, np.nan)

        for i in range(1, max_steps + 1):
            t = i * step_size

            sample_x = robot_position[0] + world_dx[active] * t
            sample_y = robot_position[1] + world_dy[active] * t
            ray_z = self.mounting_height + world_dz[active] * t

            sample_xy = np.stack((sample_x, sample_y), axis=-1)
            terrain_heights = terrain_query_fn(sample_xy)

            hits = ray_z <= terrain_heights
            
            if np.any(hits):
                active_indices = np.where(active)[0]
                hit_indices = active_indices[hits]

                hit_points[hit_indices, 0] = sample_x[hits]
                hit_points[hit_indices, 1] = sample_y[hits]
                hit_points[hit_indices, 2] = terrain_heights[hits]
                hit_depths[hit_indices] = t

                active[hit_indices] = False

            if not np.any(active):
                break

        valid_mask = ~np.isnan(hit_depths)
        return hit_points[valid_mask], hit_depths[valid_mask]
    
    def sample_terrain(self, ground_points: np.ndarray, terrain_query_fn: Callable) -> np.ndarray:
        if len(ground_points) == 0:
            return np.array([])

        sample_xy = ground_points[:, :2]
        terrain_heights = terrain_query_fn(sample_xy)
        return terrain_heights
    
    def add_noise(self, points_3d: np.ndarray, depths: np.ndarray, sigma: float) -> np.ndarray:
        noisy_points = points_3d.copy()
        sigma_lateral = sigma*depths*self.pixel_size[0] / self.focal_length
        
        fx = self.focal_length / self.pixel_size[0]
        sigma_depth = sigma * (depths ** 2) / (fx * self.baseline)

        noisy_points[:, 0] += np.random.normal(0, sigma_lateral)
        noisy_points[:, 1] += np.random.normal(0, sigma_lateral)

        incidence_angle = np.arcsin(np.clip(self.mounting_height / depths, -1, 1))
        sigma_z = sigma_depth * np.sin(incidence_angle)

        noisy_points[:, 2] += np.random.normal(0, sigma_z)

        return noisy_points
    
    def compute_point_uncertainty(self, depths: np.ndarray, sigma: float) -> dict:
        fx = self.focal_length / self.pixel_size[0]
        sigma_depth = sigma * (depths ** 2) / (fx * self.baseline)

        incidence_angle = np.arcsin(np.clip(self.mounting_height / depths, -1, 1))
        sigma = sigma_depth * np.sin(incidence_angle)

        patch_size = depths * self.pixel_size[0] / self.focal_length

        return {
            'sigma': sigma,
            'patch_size': patch_size
        }
    
    def get_point_cloud(self, robot_position: np.ndarray, robot_heading: float, terrain_query_fn: Callable,
                        noise_sigma: float = 0.5) -> dict:
        
        pixels = self.generate_pixel_grid()
        rays = self.pixel_to_rays(pixels)
        ground_points, depths = self.rays_to_ground(rays, robot_position, robot_heading, terrain_query_fn)
        noisy_points = self.add_noise(ground_points, depths, noise_sigma)
        uncertainty = self.compute_point_uncertainty(depths, noise_sigma)

        return {
            'points': noisy_points,
            'sigma': uncertainty['sigma'],
            'patch_size': uncertainty['patch_size']
        }