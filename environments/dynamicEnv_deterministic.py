import numpy as np
from typing import List, Tuple, Optional

# In this environment, we will treat all obstacles to be circles
class Obstacle:
    def __init__(self, position: np.ndarray, radius: float, velocity: np.ndarray):
        self.position = np.asarray(position, dtype=np.float32)
        self.velocity = np.asarray(velocity, dtype=np.float32)
        self.radius = radius

        self.area = np.pi*radius**2

    def move(self, dt: float):
        self.position += self.velocity*dt

class DeterministicEnv:
    def __init__(self, bounds: Optional[Tuple[float, float, float, float]] = None, robot_radius: float=0.5):
        self.bounds = bounds
        self.robot_radius = robot_radius
        self.obstacles: List[Obstacle] = []

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)

    def remove_obstacle(self, index: int) -> None:
        if 0 <= index < len(self.obstacles):
            self.obstacles.pop(index)

    def get_obstacle_data(self):
        """
        Returns obstacle data in the format expected by MPPIBaseline.
        DeterministicEnv supports circles only.
        """

        positions = []
        radii = []

        for obs in self.obstacles:
            positions.append(obs.position)
            radii.append(obs.radius)

        circles = {
            'count': len(self.obstacles),
            'positions': np.array(positions, dtype=np.float32),
            'radii': np.array(radii, dtype=np.float32)
        }

        # No rectangles or polygons in deterministic environment
        rectangles = {
            'count': 0,
            'positions': np.zeros((0, 2), dtype=np.float32),
            'widths': np.zeros(0, dtype=np.float32),
            'heights': np.zeros(0, dtype=np.float32),
            'angles': np.zeros(0, dtype=np.float32)
        }

        polygons = {
            'count': 0,
            'vertices_flat': np.zeros((0, 2), dtype=np.float32),
            'starts': np.zeros(0, dtype=np.int32),
            'lengths': np.zeros(0, dtype=np.int32)
        }

        return {
            'circles': circles,
            'rectangles': rectangles,
            'polygons': polygons
        }

    
    def move_obstacles(self, dt: float):
        if self.bounds is None:
            for obs in self.obstacles:
                obs.move(dt)
            return
        
        xmin, xmax, ymin, ymax = self.bounds
        
        for obs in self.obstacles:
            obs.move(dt)

            x, y = obs.position
            r = obs.radius

            if x-r < xmin or x+r > xmax:
                obs.velocity[0] *= -1
                obs.position[0] = np.clip(obs.position[0], xmin+r, xmax-r)
            
            if y-r < ymin or y+r > ymax:
                obs.velocity[1] *= -1
                obs.position[1] = np.clip(obs.position[1], ymin+r, ymax-r)

    def check_for_collision(self, position: np.ndarray) -> bool:
        if not self._in_bounds(position):
            return True
        
        for obs in self.obstacles:
            dist = np.linalg.norm(position[:2] - obs.position)
            if dist <= obs.radius + self.robot_radius:
                return True
            
        return False
    
    def _in_bounds(self, position: np.ndarray) -> bool:
        pos = position[:2]
        xmin, xmax, ymin, ymax = self.bounds
        if pos[0] < xmin or pos[0] > xmax or pos[1] < ymin or pos[1] > ymax:
            return False
        
        return True
    
    def get_nearest_obstacle_distance(self, position: np.ndarray) -> float:
        """
        Get distance to nearest obstacle (negative if inside obstacle).
        Useful for cost functions in trajectory optimization.
        """
        min_distance = float('inf')
        
        for obstacle in self.obstacles:
            dist = np.linalg.norm(position[:2]-obstacle.position)
            min_distance = min(min_distance, dist-obstacle.radius-self.robot_radius)
        
        return min_distance
    
    def obstacle_collisions(self):
        n_obs = len(self.obstacles)

        for i in range(n_obs):
            for j in range(i+1, n_obs):

                obs_i = self.obstacles[i]
                obs_j = self.obstacles[j]

                diff = obs_i.position - obs_j.position
                dist = np.linalg.norm(diff)
                min_dist = obs_i.radius + obs_j.radius

                if dist < min_dist and dist > 1e-6:
                    normal = diff/dist

                    v_relative = obs_i.velocity - obs_j.velocity
                    v_normal = np.dot(v_relative, normal)

                    if v_normal < 0:
                        restitution = 0.6 # Inelasticity

                        impulse = -(1+restitution)*v_normal
                        impulse /= (1.0/(obs_i.area) + 1.0/(obs_j.area))
                        
                        impulse_vector = impulse*normal

                        obs_i.velocity += impulse_vector/obs_i.area
                        obs_j.velocity -= impulse_vector/obs_j.area

                    overlap = min_dist - dist
                    if overlap > 0:
                        correction = overlap/((1.0/obs_i.area) + (1.0/obs_j.area))*normal

                        obs_i.position += correction/obs_i.area
                        obs_j.position -= correction/obs_j.area
    
    def predict_obstacle_trajectories(self, horizon, dt):
        if len(self.obstacles) == 0:
            return np.zeros((horizon, 0, 2), dtype=np.float32)

        positions = np.array(
            [obs.position.copy() for obs in self.obstacles],
            dtype=np.float32
        )
        velocities = np.array(
            [obs.velocity.copy() for obs in self.obstacles],
            dtype=np.float32
        )
        radii = np.array([obs.radius for obs in self.obstacles], dtype=np.float32)

        xmin, xmax, ymin, ymax = self.bounds

        traj = np.zeros((horizon, len(self.obstacles), 2), dtype=np.float32)

        for k in range(horizon):
            positions += velocities * dt

            # X reflection
            hit_x = (positions[:, 0] - radii < xmin) | \
                    (positions[:, 0] + radii > xmax)
            velocities[hit_x, 0] *= -1
            positions[:, 0] = np.clip(positions[:, 0], xmin + radii, xmax - radii)

            # Y reflection
            hit_y = (positions[:, 1] - radii < ymin) | \
                    (positions[:, 1] + radii > ymax)
            velocities[hit_y, 1] *= -1
            positions[:, 1] = np.clip(positions[:, 1], ymin + radii, ymax - radii)

            traj[k] = positions

        return traj

    
    def get_visualization_data(self):
        return {
            "bounds": self.bounds,
            "obstacles": [
                {
                    "position": obs.position.copy(),
                    "radius": obs.radius
                }
                for obs in self.obstacles
            ]
        }
    
    def step(self, dt: float):
        self.move_obstacles(dt)
        self.obstacle_collisions()

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

class Visualizer:
    def __init__(self, env: DeterministicEnv):
        self.env = env

        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal")

        xmin, xmax, ymin, ymax = env.bounds
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

        # Draw boundary box
        self.ax.add_patch(
            Rectangle((xmin, ymin),
                      xmax - xmin,
                      ymax - ymin,
                      fill=False,
                      linewidth=2)
        )

        # Obstacle patches
        self.obstacle_patches = []

        for obs in env.obstacles:
            circle = Circle(obs.position, obs.radius, color="red")
            self.ax.add_patch(circle)
            self.obstacle_patches.append(circle)

        # Robot patch (optional)
        self.robot_patch = Circle((0, 0), env.robot_radius, color="blue")
        self.ax.add_patch(self.robot_patch)

        plt.ion()
        plt.show()

    def render(self, robot_position=None):
        # Update obstacle positions
        for patch, obs in zip(self.obstacle_patches, self.env.obstacles):
            patch.center = obs.position

        # Update robot position
        if robot_position is not None:
            self.robot_patch.center = robot_position[:2]

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

