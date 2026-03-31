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
                if not observed[r, c]:
                    elevation[r, c] = z_new
                    precision[r, c] = sigma_new
                    confidence[r, c] = conf_point
                    observed[r, c] = True
                else:
                    prec_old = precision[r, c]
                    conf_old = confidence[r, c]
                    z_old = elevation[r, c]

                    conf_total = conf_old + conf_point
                    if conf_total == 0:
                        continue

                    elevation[r, c] = (conf_old * z_old + conf_point * z_new) / conf_total
                    precision[r, c] = (conf_old * prec_old + conf_point * sigma_new) / conf_total
                    confidence[r, c] = conf_total


@njit
def _compute_slope(rows, cols, elevation, observed, cell_size):
    slope = np.full((rows, cols), np.inf, dtype=np.float32)

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if not (observed[r, c] and
                    observed[r-1, c] and observed[r+1, c] and
                    observed[r, c-1] and observed[r, c+1]):
                continue

            dz_dx = (elevation[r, c+1] - elevation[r, c-1]) / (2 * cell_size)
            dz_dy = (elevation[r+1, c] - elevation[r-1, c]) / (2 * cell_size)
            slope[r, c] = np.sqrt(dz_dx**2 + dz_dy**2)

    return slope


@njit
def _compute_roughness(window, rows, cols, elevation, observed):
    roughness = np.full((rows, cols), np.inf, dtype=np.float32)

    for r in range(window, rows - window):
        for c in range(window, cols - window):
            patch = elevation[r-window:r+window+1, c-window:c+window+1]
            obs_patch = observed[r-window:r+window+1, c-window:c+window+1]

            if not np.all(obs_patch):
                continue

            roughness[r, c] = np.var(patch)

    return roughness


class DEMBuilder:
    def __init__(self, grid_size: Tuple[int, int], cell_size: float, origin: np.ndarray):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.origin = origin

        # the three arrays that define the DEM
        self.elevation = np.zeros(grid_size, dtype=np.float32)
        self.precision = np.full(grid_size, np.inf, dtype=np.float32)
        self.confidence = np.zeros(grid_size, dtype=np.float32)

        # track which cells have ever been observed
        self.observed = np.zeros(grid_size, dtype=bool)

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

        # weights — tune these to your robot
        w_slope = 5.0
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
        cost[~self.observed] = 100.0

        return cost
