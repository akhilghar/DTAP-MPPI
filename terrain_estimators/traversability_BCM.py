import numpy as np
from numba import njit
from typing import Optional

@njit
def _compute_attribute_vector(points, n_points, expected_density):
    """Compute the attribute vector for a given point cloud.
    
    The 8-dimensional attribute vector consists of the following components:
    1: Cell Density
    2: Variance of Elevation
    3: Mean Surface Normal in x direction
    4: Mean Surface Normal in y direction
    5: Mean Surface Normal in z direction
    6: Variance of Surface Normal in x direction
    7: Variance of Surface Normal in y direction
    8: Variance of Surface Normal in z direction"""

    attr = np.zeros(8, dtype=np.float32)
    if n_points < 3:
        attr[0] = n_points / max(expected_density, 1.0)
        attr[1] = 0.0
        attr[2] = 0.0
        attr[3] = 0.0
        attr[4] = 1.0
        attr[5] = 0.0
        attr[6] = 0.0
        attr[7] = 0.0
        return attr
    
    # Cell density
    attr[0] = n_points / max(expected_density, 1.0)

    # Elevation variance
    mean_z = 0.0
    for i in range(n_points):
        mean_z += points[i, 2]
    mean_z /= n_points
    var_z = 0.0
    for i in range(n_points):
        dz = points[i, 2] - mean_z
        var_z += dz * dz
    var_z /= n_points
    attr[1] = var_z

    # Surface normals via PCA
    cx = 0.0
    cy = 0.0
    cz = 0.0
    for i in range(n_points):
        cx += points[i, 0]
        cy += points[i, 1]
        cz += points[i, 2]
    cx /= n_points
    cy /= n_points
    cz /= n_points

    # build covariance matrix
    cov_xx = 0.0; cov_xy = 0.0; cov_xz = 0.0
    cov_yy = 0.0; cov_yz = 0.0;
    cov_zz = 0.0

    for i in range(n_points):
        dx = points[i, 0] - cx
        dy = points[i, 1] - cy
        dz = points[i, 2] - cz

        cov_xx += dx * dx
        cov_xy += dx * dy
        cov_xz += dx * dz
        cov_yy += dy * dy
        cov_yz += dy * dz
        cov_zz += dz * dz
    
    cov_xx /= n_points
    cov_xy /= n_points
    cov_xz /= n_points
    cov_yy /= n_points
    cov_yz /= n_points
    cov_zz /= n_points

    # Obtain approximate normal via cross product of eigenvectors
    if cov_xx > 1e-10 and cov_yy > 1e-10:
        dzdx = cov_xz / cov_xx
        dzdy = cov_yz / cov_yy
    else:
        dzdx = 0.0
        dzdy = 0.0

    nx = -dzdx
    ny = -dzdy
    nz = 1.0
    norm = np.sqrt(nx*nx + ny*ny + nz*nz)
    if norm > 1e-10:
        nx /= norm
        ny /= norm
        nz /= norm

    attr[2] = nx
    attr[3] = ny
    attr[4] = nz

    # Variance of surface normals (roughness)
    sum_nx = 0.0; sum_ny = 0.0; sum_nz = 0.0
    sum_nx2 = 0.0; sum_ny2 = 0.0; sum_nz2 = 0.0
    n_local = 0

    for i in range(n_points):
        if i < 2:
            continue

        deltax = points[i, 0] - points[i-1, 0]
        deltay = points[i, 1] - points[i-1, 1]
        deltaz = points[i, 2] - points[i-1, 2]
        deltax2 = points[i, 0] - points[i-2, 0]
        deltay2 = points[i, 1] - points[i-2, 1]
        deltaz2 = points[i, 2] - points[i-2, 2]

        locnorm_x = deltay * deltaz2 - deltaz * deltay2
        locnorm_y = deltaz * deltax2 - deltax * deltaz2
        locnorm_z = deltax * deltay2 - deltay * deltax2
        locnorm_len = np.sqrt(locnorm_x*locnorm_x + locnorm_y*locnorm_y + locnorm_z*locnorm_z)
        if locnorm_len > 1e-10:
            locnorm_x /= locnorm_len
            locnorm_y /= locnorm_len
            locnorm_z /= locnorm_len
            if locnorm_z < 0:
                locnorm_x = -locnorm_x
                locnorm_y = -locnorm_y
                locnorm_z = -locnorm_z

            sum_nx += locnorm_x
            sum_ny += locnorm_y
            sum_nz += locnorm_z
            sum_nx2 += locnorm_x * locnorm_x
            sum_ny2 += locnorm_y * locnorm_y
            sum_nz2 += locnorm_z * locnorm_z
            n_local += 1

    if n_local > 0:
        mean_nx = sum_nx / n_local
        mean_ny = sum_ny / n_local
        mean_nz = sum_nz / n_local

        var_nx = sum_nx2 / n_local - mean_nx * mean_nx
        var_ny = sum_ny2 / n_local - mean_ny * mean_ny
        var_nz = sum_nz2 / n_local - mean_nz * mean_nz

        attr[5] = var_nx
        attr[6] = var_ny
        attr[7] = var_nz
    else:
        # If we don't have enough local points to compute normals, set roughness to 0
        attr[5] = 0.0
        attr[6] = 0.0
        attr[7] = 0.0

    return attr

class TraversabilityClassifier:
    def __init__(self, n_classes: int=3, n_attributes: int=8, 
                 buffer_size: int=5000, retrain_interval: int=100,
                 pitch_limit: float=20.0, roll_limit: float=20.0,
                 slip_limit: float=0.5):
        self.n_classes = n_classes
        self.n_attributes = n_attributes
        self.buffer_size = buffer_size
        self.retrain_interval = retrain_interval
        self.pitch_limit = pitch_limit * np.pi / 180.0
        self.roll_limit = roll_limit * np.pi / 180.0
        self.slip_limit = slip_limit

        self.attr_buffer = np.zeros((buffer_size, n_attributes), dtype=np.float32)
        self.label_buffer = np.zeros(buffer_size, dtype=np.int32)
        self.buffer_index = 0
        self.buffer_count = 0

        self.class_means = None
        self.class_vars = None
        self.class_priors = None
        self.trained = False

        self.steps_since_retrain = 0

    def get_attribute_vector(self, points: np.ndarray, n_points: int, expected_density: float, confidence: float=1.0) -> np.ndarray:
        attributes = _compute_attribute_vector(points, n_points, expected_density)
        attributes[0] = confidence
        return attributes
    
    def label_from_dynamics(self, pitch: float, roll: float,
                            desired_vel: float, actual_vel: float) -> int:
        
        pitch_safety = abs(pitch)/self.pitch_limit
        roll_safety = abs(roll)/self.roll_limit

        if desired_vel > 0:
            slip = max(0.0, (desired_vel - actual_vel) / desired_vel)
        else:
            slip = 0.0

        slip_safety = slip / self.slip_limit

        worst_case = max(pitch_safety, roll_safety, slip_safety)
        if worst_case > 1.0:
            return 2 # Non-traversable
        elif worst_case > 0.6:
            return 1 # Cautious
        else:
            return 0 # Traversable
        
    def record_experience(self, attributes: np.ndarray, pitch: float, roll: float,
                          desired_vel: float, actual_vel: float):
        label = self.label_from_dynamics(pitch, roll, desired_vel, actual_vel)
        self.attr_buffer[self.buffer_index] = attributes
        self.label_buffer[self.buffer_index] = label
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.buffer_count = min(self.buffer_count + 1, self.buffer_size)
        self.steps_since_retrain += 1

        if (self.steps_since_retrain >= self.retrain_interval) and (self.buffer_count >= 50):
            self.fit()
            self.steps_since_retrain = 0

    def heightmap_bootstrap(self, heightmap: np.ndarray, cell_size: float,
                            patch_size: int, sample_size: int=3000,
                            noise_sigma: float=0.05):
        rows, cols = heightmap.shape
        rng = np.random.default_rng(42)

        for _ in range(sample_size):
            c = rng.integers(patch_size, cols - patch_size)
            r = rng.integers(patch_size, rows - patch_size)
            # convert 2D heightmap patch to 3D points
            patch_2d = heightmap[r-patch_size:r+patch_size+1, c-patch_size:c+patch_size+1]
            patch_rows, patch_cols = patch_2d.shape
            points = np.zeros((patch_rows * patch_cols, 3), dtype=np.float32)
            idx = 0
            for pr in range(patch_rows):
                for pc in range(patch_cols):
                    points[idx, 0] = (c - patch_size + pc) * cell_size
                    points[idx, 1] = (r - patch_size + pr) * cell_size
                    points[idx, 2] = patch_2d[pr, pc]
                    idx += 1

            # Add sensor-like noise so the classifier learns noisy-but-flat = traversable
            points[:, 2] += rng.normal(0.0, noise_sigma, size=len(points)).astype(np.float32)

            attributes = _compute_attribute_vector(
                points, len(points), float(len(points))
            )
            # Heuristic labeling based on slope and roughness
            dzdx = (patch_2d[patch_size, patch_size+1] - patch_2d[patch_size, patch_size-1]) / (2 * cell_size)
            dzdy = (patch_2d[patch_size+1, patch_size] - patch_2d[patch_size-1, patch_size]) / (2 * cell_size)
            gen_pitch = np.arctan(dzdy)
            gen_roll = np.arctan(dzdx)

            slope_mag = np.sqrt(dzdx**2 + dzdy**2)
            gen_slip = max(0.0, slope_mag-0.3) / 0.5

            v_cmd = 1.0
            v_actual = v_cmd * max(0.0, 1.0 - min(gen_slip, 0.95))
            
            self.record_experience(attributes, gen_pitch, gen_roll, v_cmd, v_actual)

        self.fit()
        self.get_class_dist()

    # Model fitting based on Gaussian Naive Bayes
    def fit(self):
        n = self.buffer_count
        attributes = self.attr_buffer[:n]
        labels = self.label_buffer[:n]

        self.class_means = np.zeros((self.n_classes, self.n_attributes), dtype=np.float32)
        self.class_vars = np.zeros((self.n_classes, self.n_attributes), dtype=np.float32)
        self.class_priors = np.zeros(self.n_classes, dtype=np.float32)

        for c in range(self.n_classes):
            mask = (labels == c)
            if np.sum(mask) < 5:
                self.class_means[c] = 0.0
                self.class_vars[c] = 1.0
                self.class_priors[c] = 1.0 / self.n_classes
                continue

            class_attrs = attributes[mask]
            self.class_means[c] = np.mean(class_attrs, axis=0)
            self.class_vars[c] = np.var(class_attrs, axis=0) + 1e-6
            self.class_priors[c] = np.sum(mask) / n

        self.trained = True

    # Predict posterior probabilities for each class given an attribute vector
    def predict(self, attributes: np.ndarray) -> np.ndarray:
        if attributes.ndim == 1:
            attributes = attributes.reshape(1, -1)
        
        if not self.trained:
            if attributes.ndim == 1:
                return np.full((attributes.shape[0], self.n_classes), 1.0 / self.n_classes, dtype=np.float32)

        n = attributes.shape[0]
        log_probs = np.zeros((n, self.n_classes), dtype=np.float32)

        for c in range(self.n_classes):
            # Use log likelihood for numerical stability - log likelihood is derived from log of Gaussian PDF
            diff = attributes - self.class_means[c]
            log_likelihood = -0.5 * np.sum((diff**2)/self.class_vars[c] + np.log(2*np.pi*self.class_vars[c]), axis=1)
            log_posterior = log_likelihood + np.log(self.class_priors[c])
            log_probs[:, c] = log_posterior

        # Normalize to get probabilities via softmax
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs
    
    # Compute a traversability score in [0,1] where more is more traversable
    def score(self, attributes: np.ndarray) -> np.ndarray:
        probs = self.predict(attributes)
        # Score is weighted sum of class probabilities (0*traversable + 0.5*cautious + 1*non-traversable)
        score = 0.5*probs[:,1] + 1.0*probs[:,2]
        return score
    
    # Get the class distribution (for analysis/visualization)
    def get_class_dist(self):
        n = self.buffer_count
        labels = self.label_buffer[:n]
        names = ['Traversable', 'Cautious', 'Non-traversable']
        for c in range(self.n_classes):
            count = np.sum(labels == c)
            print(f"{names[c]}: {count} samples ({100.0*count/n:.1f}%)")


    def get_diagnostics(self) -> dict:
        return {
            "trained": self.trained,
            "buffer_count": self.buffer_count,
            "class_priors": (
                self.class_priors.copy()
                if self.class_priors is not None
                else None
            ),
            "class_means": (
                self.class_means.copy()
                if self.class_means is not None
                else None
            ),
        }