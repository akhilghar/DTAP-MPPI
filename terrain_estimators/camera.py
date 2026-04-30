import numpy as np
from typing import Tuple, Callable
from numba import cuda, njit
from terrain_estimators.estimator_kernels import ray_to_terrain_kernel
from terrain_estimators.traversability_BCM import _compute_attribute_vector

@njit
def _batch_attribute_vectors(sorted_pts, sorted_dep, boundaries, n_cells):
    attr_matrix = np.zeros((n_cells, 8), dtype=np.float32)
    for i in range(n_cells):
        start = boundaries[i]
        end   = boundaries[i + 1]
        n_pts = end - start
        if n_pts == 0:
            continue
        cell_pts = sorted_pts[start:end]
        mean_dep  = 0.0
        for k in range(n_pts):
            mean_dep += sorted_dep[start + k]
        mean_dep /= n_pts
        expected_density = max(1.0, 20.0 * (2.0 / max(mean_dep, 0.1)) ** 2)
        attr_matrix[i] = _compute_attribute_vector(cell_pts, n_pts, expected_density)
    return attr_matrix

@njit
def _batch_centers(sorted_pts, boundaries, n_cells):
    centers = np.zeros((n_cells, 2), dtype=np.float32)
    for i in range(n_cells):
        start = boundaries[i]
        end   = boundaries[i + 1]
        if end == start:
            continue
        sx = 0.0; sy = 0.0
        for k in range(start, end):
            sx += sorted_pts[k, 0]
            sy += sorted_pts[k, 1]
        n = end - start
        centers[i, 0] = sx / n
        centers[i, 1] = sy / n
    return centers


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

        self.pixel_step = 4
        self.n_rays = (image_size[0] // self.pixel_step) * (image_size[1] // self.pixel_step)

        self.d_rays_dx = cuda.device_array(self.n_rays, dtype=np.float32)
        self.d_rays_dy = cuda.device_array(self.n_rays, dtype=np.float32)
        self.d_rays_dz = cuda.device_array(self.n_rays, dtype=np.float32)
        self.d_out_x = cuda.device_array(self.n_rays, dtype=np.float32)
        self.d_out_y = cuda.device_array(self.n_rays, dtype=np.float32)
        self.d_out_z = cuda.device_array(self.n_rays, dtype=np.float32)
        self.d_out_depth = cuda.device_array(self.n_rays, dtype=np.float32)
        self.d_out_valid = cuda.device_array(self.n_rays, dtype=np.bool_)

        self.h_out_x = np.empty(self.n_rays, dtype=np.float32)
        self.h_out_y = np.empty(self.n_rays, dtype=np.float32)
        self.h_out_z = np.empty(self.n_rays, dtype=np.float32)
        self.h_out_depth = np.empty(self.n_rays, dtype=np.float32)
        self.h_out_valid = np.empty(self.n_rays, dtype=np.bool_)

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
    
    def get_point_cloud(self, robot_position: np.ndarray, robot_heading: float, 
                        d_heightmap,
                        heightmap_origin: np.ndarray, heightmap_cell_size: float,
                        noise_sigma: float = 0.2) -> dict:
        
        pixels = self.generate_pixel_grid()
        rays = self.pixel_to_rays(pixels)
        
        theta = robot_heading  # already in radians
        world_dx = (rays[:, 2] * np.cos(theta) + rays[:, 0] * np.sin(theta)).astype(np.float32)
        world_dy = (rays[:, 2] * np.sin(theta) - rays[:, 0] * np.cos(theta)).astype(np.float32)
        world_dz = rays[:, 1].astype(np.float32)

        self.d_rays_dx.copy_to_device(world_dx)
        self.d_rays_dy.copy_to_device(world_dy)
        self.d_rays_dz.copy_to_device(world_dz)

        cam_x = np.float32(robot_position[0])
        cam_y = np.float32(robot_position[1])
        cam_z = np.float32(self.mounting_height)

        threads_per_block = 256
        blocks_per_grid = (self.n_rays + threads_per_block - 1) // threads_per_block

        ray_to_terrain_kernel[blocks_per_grid, threads_per_block](
            self.d_rays_dx, self.d_rays_dy, self.d_rays_dz, cam_x, cam_y, cam_z, d_heightmap,
            heightmap_origin[0], heightmap_origin[1], heightmap_cell_size, self.max_range, 0.1,
            self.d_out_x, self.d_out_y, self.d_out_z, self.d_out_depth, self.d_out_valid
        )

        self.d_out_x.copy_to_host(self.h_out_x)
        self.d_out_y.copy_to_host(self.h_out_y)
        self.d_out_z.copy_to_host(self.h_out_z)
        self.d_out_depth.copy_to_host(self.h_out_depth)
        self.d_out_valid.copy_to_host(self.h_out_valid)

        valid_points = self.h_out_valid
        points_3d = np.stack((self.h_out_x[valid_points], self.h_out_y[valid_points], self.h_out_z[valid_points]), axis=-1)
        depths = self.h_out_depth[valid_points]

        noisy_points = self.add_noise(points_3d, depths, noise_sigma)
        uncertainty = self.compute_point_uncertainty(depths, noise_sigma)

        return {
            'points': noisy_points,
            'depths': depths,
            'sigma': uncertainty['sigma'],
            'patch_size': uncertainty['patch_size']
        }
    
    def classify_point_cloud(self, point_cloud: dict, classifier, cell_size: float) -> np.ndarray:
        points = point_cloud['points']
        depths = point_cloud['depths']

        xmin, ymin = points[:, 0].min(), points[:, 1].min()
        n_cols = max(1, int(np.ceil((points[:, 0].max() - xmin) / cell_size)))
        n_rows = max(1, int(np.ceil((points[:, 1].max() - ymin) / cell_size)))

        col_idx = np.clip((points[:, 0] - xmin) / cell_size, 0, n_cols - 1).astype(int)
        row_idx = np.clip((points[:, 1] - ymin) / cell_size, 0, n_rows - 1).astype(int)
        cell_ids = row_idx * n_cols + col_idx

        unique_cells, inverse = np.unique(cell_ids, return_inverse=True)
        n_cells = len(unique_cells)

        # Sort points by cell so each cell's points are contiguous
        sort_idx = np.argsort(inverse, kind='stable')
        sorted_inv = inverse[sort_idx]
        sorted_pts = points[sort_idx]
        sorted_dep = depths[sort_idx]

        # Cell boundaries in the sorted array
        boundaries = np.searchsorted(sorted_inv, np.arange(n_cells + 1))

        # Build per-cell attribute matrix and centers in one vectorized pass
        attr_matrix = _batch_attribute_vectors(sorted_pts, sorted_dep, boundaries, n_cells)
        centers = _batch_centers(sorted_pts, boundaries, n_cells)

        probs = classifier.predict(attr_matrix)
        scores = (0.5 * probs[:, 1] + 1.0 * probs[:, 2]).ravel()
        labels = np.argmax(probs, axis=1).astype(np.int8)
        return scores.astype(np.float32), centers, labels