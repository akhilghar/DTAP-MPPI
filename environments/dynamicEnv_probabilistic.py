import numpy as np
from enum import Enum
from typing import List, Tuple, Optional


class ObstacleMode(Enum):
    APATHETIC = "apathetic"   # Random wandering, ignores robot
    AVOIDANT  = "avoidant"    # Actively steers away from robot
    STATIC  = "static"     # No movement, purely a static obstacle (can be implemented as a special case of apathetic with zero speed)


# In this environment, we will treat all obstacles to be circles
class Obstacle:

    def __init__(self, position: np.ndarray, radius: float, velocity: np.ndarray = None, mode: ObstacleMode = ObstacleMode.APATHETIC,
        speed: float = None, tau: float = 0.1, avoidance_radius: float = 3.0, avoidance_strength: float = 3.0,
        neighbor_avoidance_radius: float = 1.5, neighbor_avoidance_strength: float = 1.0):
        self.position = np.asarray(position, dtype=np.float32)
        if velocity is None:
            velocity = np.zeros_like(self.position)
        self.velocity = np.asarray(velocity, dtype=np.float32)
        self.radius = radius
        self.area   = np.pi * radius ** 2

        # Desired speed: infer from initial velocity or default to 1 m/s
        self.speed = speed if speed is not None else (float(np.linalg.norm(self.velocity)) or 1.0)

        self.mode = mode

        # tau: velocity relaxation time — larger = more sluggish / inertial
        self.tau = tau

        # Avoidant mode parameters (robot)
        self.avoidance_radius   = avoidance_radius    # perceive robot within this distance
        self.avoidance_strength = avoidance_strength  # repulsion gain

        # Neighbor avoidance parameters (other obstacles)
        # neighbor_avoidance_radius is a surface-gap threshold (center dist minus combined radii)
        self.neighbor_avoidance_radius   = neighbor_avoidance_radius
        self.neighbor_avoidance_strength = neighbor_avoidance_strength

        # Wandering goal (used by apathetic mode and as a fallback in avoidant mode)
        self._goal: Optional[np.ndarray] = None
        self._goal_timer    = 0.0
        self._goal_duration = np.random.uniform(3.0, 8.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_goal(self, bounds=None) -> None:
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            self._goal = np.array([
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax),
            ], dtype=np.float32)
        else:
            angle = np.random.uniform(0, 2 * np.pi)
            dist  = np.random.uniform(2.0, 6.0)
            self._goal = (self.position + dist * np.array(
                [np.cos(angle), np.sin(angle)], dtype=np.float32))
        self._goal_timer    = 0.0
        self._goal_duration = np.random.uniform(3.0, 8.0)

    def _desired_velocity_apathetic(self, bounds=None) -> np.ndarray:
        """Drift toward a slowly-refreshing random goal with naturalistic jitter."""
        if self._goal is None or self._goal_timer >= self._goal_duration:
            self._refresh_goal(bounds)

        to_goal = self._goal - self.position
        dist    = np.linalg.norm(to_goal)
        if dist < 0.3:
            self._refresh_goal(bounds)
            to_goal = self._goal - self.position
            dist    = np.linalg.norm(to_goal)

        direction = to_goal / (dist + 1e-6)
        # Small Gaussian noise for naturalistic variation
        direction = direction + np.random.randn(2).astype(np.float32) * 0.1
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm

        return self.speed * direction

    def _desired_velocity_avoidant(self, robot_pos: Optional[np.ndarray], bounds=None) -> np.ndarray:
        """
        Steer away from the robot with strength proportional to 1/dist².
        Blends with the apathetic wander near the edge of perception so
        motion stays naturalistic rather than purely reactive.
        Falls back to apathetic wandering when robot is unperceived.
        """
        if robot_pos is None:
            return self._desired_velocity_apathetic(bounds)

        to_robot = robot_pos[:2] - self.position
        dist     = np.linalg.norm(to_robot)

        if dist > self.avoidance_radius or dist < 1e-6:
            return self._desired_velocity_apathetic(bounds)

        # Repulsion direction (directly away from robot)
        away          = -to_robot / dist
        repulsion_mag = self.avoidance_strength / (dist ** 2 + 1e-3)
        repulsion_mag = float(np.clip(repulsion_mag, 0.0, self.speed * 3.0))

        # Blend weight: 1 when robot is very close, 0 at the perception boundary
        w = 1.0 - dist / self.avoidance_radius

        wander     = self._desired_velocity_apathetic(bounds)
        wander_dir = wander / (np.linalg.norm(wander) + 1e-6)

        desired = w * repulsion_mag * away + (1.0 - w) * self.speed * wander_dir
        norm    = np.linalg.norm(desired)
        if norm > 1e-6:
            desired = desired / norm * self.speed

        return desired.astype(np.float32)

    def _desired_velocity_static(self, bounds=None) -> np.ndarray:
        """Static obstacles do not move."""
        return np.zeros(2, dtype=np.float32)

    def _neighbor_repulsion(self, neighbors: list) -> np.ndarray:
        """
        Compute a repulsion velocity contribution from nearby obstacles.
        `neighbors` is a list of (position, radius) tuples for every other obstacle.
        Repulsion is keyed on the surface gap (center distance - combined radii) so
        it is automatically proportional to obstacle size.
        """
        if self.mode != ObstacleMode.STATIC:
            repulsion = np.zeros(2, dtype=np.float32)
            for (pos, radius) in neighbors:
                diff        = self.position - pos
                dist        = np.linalg.norm(diff)
                surface_gap = dist - (self.radius + radius)
                if surface_gap < self.neighbor_avoidance_radius and dist > 1e-6:
                    direction = diff / dist
                    mag       = self.neighbor_avoidance_strength / (surface_gap ** 2 + 1e-3)
                    mag       = float(np.clip(mag, 0.0, self.speed * 3.0))
                    repulsion += mag * direction
        else:
            repulsion = np.zeros(2, dtype=np.float32)
        return repulsion

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def move(self, dt: float, robot_pos: Optional[np.ndarray] = None, bounds=None, neighbors: Optional[list] = None) -> None:
        self._goal_timer += dt

        if self.mode == ObstacleMode.APATHETIC:
            v_desired = self._desired_velocity_apathetic(bounds)
        elif self.mode == ObstacleMode.STATIC:
            v_desired = self._desired_velocity_static(bounds)
            return  # Static obstacles do not move, so we can skip the rest of the function
        else:  # AVOIDANT
            v_desired = self._desired_velocity_avoidant(robot_pos, bounds)

        # Layer in neighbor repulsion additively.
        # We don't renormalize so that a crowded obstacle can briefly exceed its
        # nominal speed to escape — the first-order steering will damp it back down.
        if neighbors:
            repulsion  = self._neighbor_repulsion(neighbors)
            v_desired  = v_desired + repulsion
            norm       = np.linalg.norm(v_desired)
            if norm > self.speed * 3.0:
                v_desired = v_desired / norm * self.speed * 3.0

        # First-order steering: smoothly relax toward desired velocity.
        # This gives the obstacle inertia, making motion feel physical.
        self.velocity += (v_desired - self.velocity) * (dt / self.tau)

        self.position += self.velocity * dt

# ====================================================================================
# ProbabilisticEnv: 2D environment with dynamic circular obstacles that can be apathetic or avoidant.
# ====================================================================================
class ProbabilisticEnv:
    def __init__(
        self,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        robot_radius: float = 0.5,
    ):
        self.bounds = bounds
        self.robot_radius = robot_radius
        self.obstacles: List[Obstacle] = []

        self.terrain = None  # Placeholder for future terrain-aware behavior
        self.dx = 0.5 # Terrain grid resolution in x direction
        self.dy = self.dx # Terrain grid resolution in y direction

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)

    def remove_obstacle(self, index: int) -> None:
        if 0 <= index < len(self.obstacles):
            self.obstacles.pop(index)

    def get_obstacle_data(self):
        """
        Returns obstacle data in the format expected by MPPIBaseline.
        ProbabilisticEnv supports circles only.
        """
        positions = []
        radii     = []

        for obs in self.obstacles:
            positions.append(obs.position)
            radii.append(obs.radius)

        circles = {
            'count':     len(self.obstacles),
            'positions': np.array(positions, dtype=np.float32),
            'radii':     np.array(radii,     dtype=np.float32),
        }

        rectangles = {
            'count':     0,
            'positions': np.zeros((0, 2), dtype=np.float32),
            'widths':    np.zeros(0,      dtype=np.float32),
            'heights':   np.zeros(0,      dtype=np.float32),
            'angles':    np.zeros(0,      dtype=np.float32),
        }

        polygons = {
            'count':         0,
            'vertices_flat': np.zeros((0, 2), dtype=np.float32),
            'starts':        np.zeros(0,      dtype=np.int32),
            'lengths':       np.zeros(0,      dtype=np.int32),
        }

        return {'circles': circles, 'rectangles': rectangles, 'polygons': polygons}

    def move_obstacles(self, dt: float, robot_pos: Optional[np.ndarray] = None) -> None:
        # Pre-build (position, radius) snapshot so each obstacle sees the same
        # neighbor state at the start of the timestep, not partially-updated positions.
        neighbor_snapshot = [(o.position.copy(), o.radius) for o in self.obstacles]

        if self.bounds is None:
            for i, obs in enumerate(self.obstacles):
                neighbors = [neighbor_snapshot[j] for j in range(len(self.obstacles)) if j != i]
                obs.move(dt, robot_pos=robot_pos, bounds=None, neighbors=neighbors)
            return

        xmin, xmax, ymin, ymax = self.bounds

        for i, obs in enumerate(self.obstacles):
            if obs.mode != ObstacleMode.STATIC:  # Static obstacles do not move, so we can skip the move call for them
                neighbors = [neighbor_snapshot[j] for j in range(len(self.obstacles)) if j != i]
                obs.move(dt, robot_pos=robot_pos, bounds=self.bounds, neighbors=neighbors)

                x, y = obs.position
                r    = obs.radius

                # Wall bounce — reflect velocity and clamp position
                if x - r < xmin or x + r > xmax:
                    obs.velocity[0] *= -1
                    obs.position[0]  = np.clip(obs.position[0], xmin + r, xmax - r)

                if y - r < ymin or y + r > ymax:
                    obs.velocity[1] *= -1
                    obs.position[1]  = np.clip(obs.position[1], ymin + r, ymax - r)

    def check_for_collision(self, position: np.ndarray) -> bool:
        if not self._in_bounds(position):
            return True

        for obs in self.obstacles:
            dist = np.linalg.norm(position[:2] - obs.position)
            if dist <= obs.radius + self.robot_radius + 0.01:  # small buffer
                return True

        return False

    def _in_bounds(self, position: np.ndarray) -> bool:
        pos              = position[:2]
        xmin, xmax, ymin, ymax = self.bounds
        if (pos[0] < xmin + self.robot_radius or pos[0] > xmax - self.robot_radius
                or pos[1] < ymin + self.robot_radius or pos[1] > ymax - self.robot_radius):
            return False
        return True

    def get_nearest_obstacle_distance(self, position: np.ndarray) -> float:
        """
        Get distance to nearest obstacle (negative if inside obstacle).
        Useful for cost functions in trajectory optimization.
        """
        min_distance = float('inf')
        for obstacle in self.obstacles:
            dist         = np.linalg.norm(position[:2] - obstacle.position)
            min_distance = min(min_distance, dist - obstacle.radius - self.robot_radius)
        return min_distance

    def obstacle_collisions(self) -> None:
        n_obs = len(self.obstacles)

        for i in range(n_obs):
            for j in range(i + 1, n_obs):
                obs_i = self.obstacles[i]
                obs_j = self.obstacles[j]

                diff     = obs_i.position - obs_j.position
                dist     = np.linalg.norm(diff)
                min_dist = obs_i.radius + obs_j.radius

                if dist < min_dist and dist > 1e-6:
                    normal = diff / dist

                    v_relative = obs_i.velocity - obs_j.velocity
                    v_normal   = np.dot(v_relative, normal)

                    if v_normal < 0:
                        restitution    = 0.6
                        impulse        = -(1 + restitution) * v_normal
                        impulse       /= 1.0 / obs_i.area + 1.0 / obs_j.area
                        impulse_vector = impulse * normal

                        obs_i.velocity += impulse_vector / obs_i.area
                        obs_j.velocity -= impulse_vector / obs_j.area

                    overlap = min_dist - dist
                    if overlap > 0:
                        correction = (overlap / (1.0 / obs_i.area + 1.0 / obs_j.area)) * normal
                        obs_i.position += correction / obs_i.area
                        obs_j.position -= correction / obs_j.area

    def predict_obstacle_trajectories(
        self,
        horizon: int,
        dt: float,
        num_rollouts: int = 50,
        direction_change_prob: float = 0.05,
    ) -> np.ndarray:
        """
        CPU Monte Carlo obstacle trajectory prediction.

        Mirrors the output format of obs_mc_rollout_kernel exactly so this can
        serve as a CPU fallback or for testing/visualisation without a GPU.

        Returns:
            all_trajs : (R, N, horizon, 2)  — all_trajs[r, n, k, xy] is the
                        position of obstacle n at step k in rollout r.
        """
        N = len(self.obstacles)
        R = num_rollouts

        if N == 0:
            return np.zeros((R, 0, horizon, 2), dtype=np.float32)

        base_pos = np.array([obs.position.copy() for obs in self.obstacles], dtype=np.float64)  # (N, 2)
        base_vel = np.array([obs.velocity.copy() for obs in self.obstacles], dtype=np.float64)  # (N, 2)
        radii    = np.array([obs.radius           for obs in self.obstacles], dtype=np.float64)  # (N,)

        xmin, xmax, ymin, ymax = self.bounds

        # Broadcast starting state across all rollouts: (R, N, 2)
        pos = np.tile(base_pos[np.newaxis], (R, 1, 1))   # (R, N, 2)
        vel = np.tile(base_vel[np.newaxis], (R, 1, 1))   # (R, N, 2)
        spd = np.linalg.norm(vel, axis=2)                # (R, N)

        all_trajs = np.zeros((R, N, horizon, 2), dtype=np.float32)

        for k in range(horizon):
            # Stochastic direction changes — each (rollout, obstacle) pair independently
            change = np.random.rand(R, N) < direction_change_prob
            angles = np.random.uniform(0, 2 * np.pi, (R, N))
            new_vx = np.cos(angles) * spd
            new_vy = np.sin(angles) * spd
            vel[:, :, 0] = np.where(change, new_vx, vel[:, :, 0])
            vel[:, :, 1] = np.where(change, new_vy, vel[:, :, 1])

            pos += vel * dt

            # Wall bouncing — vectorized over all rollouts and obstacles
            hit_x = (pos[:, :, 0] - radii < xmin) | (pos[:, :, 0] + radii > xmax)
            hit_y = (pos[:, :, 1] - radii < ymin) | (pos[:, :, 1] + radii > ymax)
            vel[:, :, 0] = np.where(hit_x, -vel[:, :, 0], vel[:, :, 0])
            vel[:, :, 1] = np.where(hit_y, -vel[:, :, 1], vel[:, :, 1])
            pos[:, :, 0] = np.clip(pos[:, :, 0], xmin + radii, xmax - radii)
            pos[:, :, 1] = np.clip(pos[:, :, 1], ymin + radii, ymax - radii)

            all_trajs[:, :, k, :] = pos          # store all R positions at step k

        return all_trajs

    def generate_terrain(self, flat: bool) -> None:
        xmin, xmax, ymin, ymax = self.bounds
        terrain_size_x = int((xmax - xmin) / self.dx)
        terrain_size_y = int((ymax - ymin) / self.dy)

        if flat:
            self.terrain = np.zeros((terrain_size_x, terrain_size_y), dtype=np.float32)  # Flat terrain has zero cost everywhere
            return

        terrain_startpos_x = int(-xmin / self.dx)
        terrain_startpos_y = int(-ymin / self.dy)
        terrain_goalpos_x = int((10.0-xmin) / self.dx) # Change this if the goal is not at (10,10)
        terrain_goalpos_y = int((10.0-ymin) / self.dy) # Change this if the goal is not at (10,10)

        self.terrain = 0.3*np.linalg.norm(np.array([self.dx, self.dy]))*np.random.randn(terrain_size_x, terrain_size_y).astype(np.float32)


    def get_visualization_data(self) -> dict:
        return {
            "bounds": self.bounds,
            "obstacles": [
                {"position": obs.position.copy(), "radius": obs.radius, "mode": obs.mode.value}
                for obs in self.obstacles
            ],
        }
    
    def get_static_obstacle_costmap(self, grid_size: Tuple[int, int], cell_size: float,
                                     origin: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        costmap = np.zeros(grid_size, dtype=np.float32)
        origin_x, origin_y = origin

        for obs in self.obstacles:
            if obs.mode == ObstacleMode.STATIC:
                # Compute the grid cells that fall within the obstacle's radius
                obs_x, obs_y = obs.position
                obs_r = obs.radius

                # Determine the bounding box of the obstacle in grid coordinates
                # (offset by origin so cell indices match the grid's coordinate frame)
                x_min = int(max(0, (obs_x - obs_r - origin_x) / cell_size))
                x_max = int(min(grid_size[0] - 1, (obs_x + obs_r - origin_x) / cell_size))
                y_min = int(max(0, (obs_y - obs_r - origin_y) / cell_size))
                y_max = int(min(grid_size[1] - 1, (obs_y + obs_r - origin_y) / cell_size))

                for x in range(x_min, x_max + 1):
                    for y in range(y_min, y_max + 1):
                        cell_center_x = origin_x + x * cell_size + cell_size / 2.0
                        cell_center_y = origin_y + y * cell_size + cell_size / 2.0
                        dist_to_obs = np.linalg.norm(np.array([cell_center_x - obs_x, cell_center_y - obs_y]))
                        if dist_to_obs <= obs_r:
                            costmap[x, y] = 100.0  # Mark as occupied

        return costmap

    def step(self, dt: float, robot_pos: Optional[np.ndarray] = None) -> None:
        self.move_obstacles(dt, robot_pos=robot_pos)
        self.obstacle_collisions()
