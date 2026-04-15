import numpy as np
from typing import Tuple, Optional
from numba import njit


@njit
def _fuse_point(points, sigmas, patch_sizes, elevation, precision, confidence,
                observed, origin_x, origin_y, cell_size, rows, cols):
    for i in range(len(points)):
        half = patch_sizes[i] / 2.0
        min_c = int(np.floor((points[i, 0] - half - origin_x) / cell_size))
        max_c = int(np.floor((points[i, 0] + half - origin_x) / cell_size))
        min_r = int(np.floor((points[i, 1] - half - origin_y) / cell_size))
        max_r = int(np.floor((points[i, 1] + half - origin_y) / cell_size))

        min_r = max(min_r, 0)
        max_r = min(max_r, rows - 1)
        min_c = max(min_c, 0)
        max_c = min(max_c, cols - 1)

        n_cells = (max_r - min_r + 1) * (max_c - min_c + 1)
        if n_cells <= 0:
            continue

        conf_point = 1.0 / n_cells
        z_new = points[i, 2]
        sigma_new = sigmas[i]

        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if not observed[c, r]:
                    elevation[c, r] = z_new
                    precision[c, r] = sigma_new
                    confidence[c, r] = conf_point
                    observed[c, r] = True
                else:
                    prec_old = precision[c, r]
                    conf_old = confidence[c, r]
                    z_old = elevation[c, r]

                    conf_total = conf_old + conf_point
                    if conf_total == 0:
                        continue

                    elevation[c, r] = (conf_old * z_old + conf_point * z_new) / conf_total
                    precision[c, r] = (conf_old * prec_old + conf_point * sigma_new) / conf_total
                    confidence[c, r] = conf_total


@njit
def _compute_slope(rows, cols, elevation, observed, cell_size):
    slope = np.zeros((cols, rows), dtype=np.float32)

    for c in range(1, cols - 1):
        for r in range(1, rows - 1):
            if not (observed[c, r] and
                    observed[c-1, r] and observed[c+1, r] and
                    observed[c, r-1] and observed[c, r+1]):
                continue

            dz_dc = (elevation[c+1, r] - elevation[c-1, r]) / (2 * cell_size)
            dz_dr = (elevation[c, r+1] - elevation[c, r-1]) / (2 * cell_size)
            slope[c, r] = np.sqrt(dz_dc**2 + dz_dr**2)

    return slope


@njit
def _compute_roughness(window, rows, cols, elevation, observed):
    roughness = np.zeros((cols, rows), dtype=np.float32)

    for c in range(window, cols - window):
        for r in range(window, rows - window):
            patch = elevation[c-window:c+window+1, r-window:r+window+1]
            obs_patch = observed[c-window:c+window+1, r-window:r+window+1]

            if not np.all(obs_patch):
                continue

            roughness[c, r] = np.var(patch)

    return roughness


class DEMBuilder:
    def __init__(self, grid_size: Tuple[int, int], cell_size: float, origin: np.ndarray):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.origin = origin

        self.z_baseline = None

        # the three arrays that define the DEM
        self.elevation = np.zeros(grid_size, dtype=np.float32)
        self.precision = np.full(grid_size, np.inf, dtype=np.float32)
        self.confidence = np.zeros(grid_size, dtype=np.float32)

        # track which cells have ever been observed
        self.observed = np.zeros(grid_size, dtype=bool)

        # Traversability cost
        self.traversability_overlay = np.full(grid_size, 0.0, dtype=np.float32)

    def world_to_grid(self, points_xy: np.ndarray) -> np.ndarray:
        grid_coords = (points_xy - self.origin) / self.cell_size
        return np.floor(grid_coords).astype(int)

    def in_bounds(self, grid_indices: np.ndarray) -> np.ndarray:
        return (
            (grid_indices[:, 0] >= 0) &
            (grid_indices[:, 0] < self.grid_size[0]) &
            (grid_indices[:, 1] >= 0) &
            (grid_indices[:, 1] < self.grid_size[1])
        )
    
    def point_in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]

    def compute_patch_cells(self, point_xy: np.ndarray, patch_size: float) -> np.ndarray:
        half = patch_size / 2.0
        min_xy = point_xy - half
        max_xy = point_xy + half

        min_grid = np.floor((min_xy - self.origin) / self.cell_size).astype(int)
        max_grid = np.floor((max_xy - self.origin) / self.cell_size).astype(int)

        # clamp to grid bounds
        min_grid = np.clip(min_grid, 0, [self.grid_size[1] - 1, self.grid_size[0] - 1])
        max_grid = np.clip(max_grid, 0, [self.grid_size[1] - 1, self.grid_size[0] - 1])

        rows = np.arange(min_grid[1], max_grid[1] + 1)
        cols = np.arange(min_grid[0], max_grid[0] + 1)
        grid = np.stack(np.meshgrid(rows, cols, indexing='ij'), axis=-1)

        return grid.reshape(-1, 2)

    def fuse_point_cloud(self, point_cloud: dict):
        points = point_cloud["points"]
        sigma = point_cloud["sigma"]
        patch_sizes = point_cloud["patch_size"]

        if self.z_baseline is None:
            self.z_baseline = float(points[:, 2].mean())

        points = points.copy()
        points[:, 2] -= self.z_baseline

        _fuse_point(points, sigma, patch_sizes,
                    self.elevation, self.precision, self.confidence, self.observed,
                    self.origin[0], self.origin[1], self.cell_size,
                    self.grid_size[0], self.grid_size[1])

    def get_elevation(self, row: int, col: int) -> Optional[float]:
        if self.observed[row, col]:
            return self.elevation[row, col]
        return None

    def get_traversability_cost(self) -> np.ndarray:
        slope = _compute_slope(self.grid_size[0], self.grid_size[1],
                               self.elevation, self.observed, self.cell_size)
        roughness = _compute_roughness(2, self.grid_size[0], self.grid_size[1],
                                       self.elevation, self.observed)

        # weights
        w_slope = 9.0
        w_rough = 10.0
        w_uncertain = 2.0

        # uncertainty penalty: high sigma or low confidence = risky
        uncertainty = np.where(
            self.confidence > 0,
            self.precision / self.confidence,
            np.inf,
        )

        cost = (
            w_slope * slope +
            w_rough * roughness +
            w_uncertain * uncertainty
        )

        # unobserved cells get a high default cost
        cost = np.where(np.isfinite(cost), cost, 50.0) 
        cost[~self.observed] = 10.0

        return cost
    
    def get_cost_at_points(self, points_xy: np.ndarray) -> np.ndarray:
        grid_coords = self.world_to_grid(points_xy)
        rows = np.clip(np.floor(grid_coords[:, 1]).astype(int), 0, self.grid_size[0] - 1)
        cols = np.clip(np.floor(grid_coords[:, 0]).astype(int), 0, self.grid_size[1] - 1)
        cost_grid = self.get_traversability_cost()
        trrn_cost = cost_grid[rows, cols]

        classification_cost = self.traversability_overlay[rows, cols]*50.0
        total_cost = trrn_cost + classification_cost
        return total_cost
