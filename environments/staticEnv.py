# environments/staticEnv.py
import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

class ObstacleType(Enum):
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    POLYGON = "polygon"

@dataclass
class Obstacle:
    """Base obstacle representation"""
    type: ObstacleType
    position: np.ndarray  # [x, y]
    
class CircleObstacle(Obstacle):
    """Circular obstacle"""
    def __init__(self, position: np.ndarray, radius: float):
        super().__init__(ObstacleType.CIRCLE, position)
        self.radius = radius
    
class RectangleObstacle(Obstacle):
    """Rectangular obstacle (axis-aligned)"""
    def __init__(self, position: np.ndarray, width: float, height: float, 
                 angle: float = 0.0):
        super().__init__(ObstacleType.RECTANGLE, position)
        self.width = width
        self.height = height
        self.angle = angle  # rotation in radians
        
class PolygonObstacle(Obstacle):
    """Arbitrary polygon obstacle"""
    def __init__(self, vertices: np.ndarray):
        # vertices shape: (N, 2)
        center = np.mean(vertices, axis=0)
        super().__init__(ObstacleType.POLYGON, center)
        self.vertices = vertices


class StaticEnvironment:
    """
    Static environment with obstacles for collision checking.
    Supports multiple obstacle types and efficient collision queries.
    """
    
    def __init__(self, bounds: Optional[Tuple[float, float, float, float]] = None,
                 robot_radius: float = 0.5):
        """
        Args:
            bounds: Environment bounds (x_min, x_max, y_min, y_max)
            robot_radius: Robot radius for collision checking (assumes circular robot)
        """
        self.bounds = bounds if bounds is not None else (-10, 10, -10, 10)
        self.robot_radius = robot_radius
        self.obstacles: List[Obstacle] = []
        
    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add an obstacle to the environment"""
        self.obstacles.append(obstacle)
    
    def add_circle_obstacle(self, position: np.ndarray, radius: float) -> None:
        """Convenience method to add circular obstacle"""
        self.add_obstacle(CircleObstacle(position, radius))
    
    def add_rectangle_obstacle(self, position: np.ndarray, width: float, 
                               height: float, angle: float = 0.0) -> None:
        """Convenience method to add rectangular obstacle"""
        self.add_obstacle(RectangleObstacle(position, width, height, angle))
    
    def add_polygon_obstacle(self, vertices: np.ndarray) -> None:
        """Convenience method to add polygon obstacle"""
        self.add_obstacle(PolygonObstacle(vertices))
    
    def remove_obstacle(self, index: int) -> None:
        """Remove obstacle by index"""
        if 0 <= index < len(self.obstacles):
            self.obstacles.pop(index)
    
    def clear_obstacles(self) -> None:
        """Remove all obstacles"""
        self.obstacles.clear()

    def get_obstacle_data(self):
        """
        Get obstacle data in a format suitable for GPU kernels.
        
        Returns:
            dict: Contains structured data for each obstacle type:
                - 'circles': dict with 'positions' (N_circles, 2) and 'radii' (N_circles,)
                - 'rectangles': dict with 'positions' (N_rects, 2), 'widths', 'heights', 'angles'
                - 'polygons': dict with 'vertices_list' (list of (N_i, 2) arrays) and 'num_vertices' (list of vertex counts)
                - 'num_obstacles': total number of obstacles
                - 'obstacle_types': list of ObstacleType enums in same order as self.obstacles
        """
        circles = {'positions': [], 'radii': []}
        rectangles = {'positions': [], 'widths': [], 'heights': [], 'angles': []}
        polygons = {'vertices_list': [], 'num_vertices': []}
        obstacle_types = []
        
        for obstacle in self.obstacles:
            obstacle_types.append(obstacle.type)
            
            if obstacle.type == ObstacleType.CIRCLE:
                circles['positions'].append(obstacle.position)
                circles['radii'].append(obstacle.radius)
                
            elif obstacle.type == ObstacleType.RECTANGLE:
                rectangles['positions'].append(obstacle.position)
                rectangles['widths'].append(obstacle.width)
                rectangles['heights'].append(obstacle.height)
                rectangles['angles'].append(obstacle.angle)
                
            elif obstacle.type == ObstacleType.POLYGON:
                polygons['vertices_list'].append(obstacle.vertices)
                polygons['num_vertices'].append(len(obstacle.vertices))
        
        # Convert lists to numpy arrays where applicable
        result = {
            'num_obstacles': len(self.obstacles),
            'obstacle_types': obstacle_types,
            'circles': {
                'positions': np.array(circles['positions'], dtype=np.float32) if circles['positions'] else np.empty((0, 2), dtype=np.float32),
                'radii': np.array(circles['radii'], dtype=np.float32) if circles['radii'] else np.empty((0,), dtype=np.float32),
                'count': len(circles['radii'])
            },
            'rectangles': {
                'positions': np.array(rectangles['positions'], dtype=np.float32) if rectangles['positions'] else np.empty((0, 2), dtype=np.float32),
                'widths': np.array(rectangles['widths'], dtype=np.float32) if rectangles['widths'] else np.empty((0,), dtype=np.float32),
                'heights': np.array(rectangles['heights'], dtype=np.float32) if rectangles['heights'] else np.empty((0,), dtype=np.float32),
                'angles': np.array(rectangles['angles'], dtype=np.float32) if rectangles['angles'] else np.empty((0,), dtype=np.float32),
                'count': len(rectangles['widths'])
            },
            'polygons': {
                'vertices_list': polygons['vertices_list'],
                'num_vertices': np.array(polygons['num_vertices'], dtype=np.int32) if polygons['num_vertices'] else np.empty((0,), dtype=np.int32),
                'count': len(polygons['num_vertices']),
                'vertices_flat': self._flatten_polygon_vertices(polygons['vertices_list']),
                'starts': self._compute_polygon_starts(polygons['num_vertices']),
                'lengths': np.array(polygons['num_vertices'], dtype=np.int32) if polygons['num_vertices'] else np.empty((0,), dtype=np.int32)
            }
        }
        
        return result
    
    def _flatten_polygon_vertices(self, vertices_list):
        """Flatten list of polygon vertex arrays into a single (N_total, 2) array"""
        if not vertices_list:
            return np.empty((0, 2), dtype=np.float32)
        flat_vertices = np.vstack(vertices_list).astype(np.float32)
        return flat_vertices
    
    def _compute_polygon_starts(self, num_vertices):
        """Compute start indices for each polygon in the flattened vertex array"""
        if not num_vertices:
            return np.empty((0,), dtype=np.int32)
        starts = np.zeros(len(num_vertices), dtype=np.int32)
        cumsum = 0
        for i, n in enumerate(num_vertices):
            starts[i] = cumsum
            cumsum += n
        return starts
    
    def check_collision(self, position: np.ndarray) -> bool:
        """
        Check if a position collides with any obstacle.
        
        Args:
            position: [x, y] position to check
            
        Returns:
            True if collision detected, False otherwise
        """
        # Check bounds
        if not self._in_bounds(position):
            return True
        
        # Check each obstacle
        for obstacle in self.obstacles:
            if self._collides_with_obstacle(position, obstacle):
                return True
        
        return False
    
    def check_trajectory_collision(self, trajectory: np.ndarray, 
                                   resolution: int = 10) -> bool:
        """
        Check if a trajectory collides with obstacles.
        
        Args:
            trajectory: Array of positions, shape (N, 2) or (N, state_dim)
                       where first 2 dimensions are x, y
            resolution: Number of points to interpolate between waypoints
            
        Returns:
            True if any collision detected
        """
        positions = trajectory[:, :2]  # Extract x, y
        
        for i in range(len(positions) - 1):
            # Interpolate between consecutive points
            for alpha in np.linspace(0, 1, resolution):
                interp_pos = (1 - alpha) * positions[i] + alpha * positions[i + 1]
                if self.check_collision(interp_pos):
                    return True
        
        return False
    
    def get_collision_info(self, position: np.ndarray) -> Tuple[bool, Optional[int], Optional[float]]:
        """
        Get detailed collision information.
        
        Returns:
            (is_collision, obstacle_index, penetration_depth)
        """
        if not self._in_bounds(position):
            return True, None, None
        
        for idx, obstacle in enumerate(self.obstacles):
            if self._collides_with_obstacle(position, obstacle):
                depth = self._penetration_depth(position, obstacle)
                return True, idx, depth
        
        return False, None, None
    
    def get_nearest_obstacle_distance(self, position: np.ndarray) -> float:
        """
        Get distance to nearest obstacle (negative if inside obstacle).
        Useful for cost functions in trajectory optimization.
        """
        min_distance = float('inf')
        
        for obstacle in self.obstacles:
            dist = self._distance_to_obstacle(position, obstacle)
            min_distance = min(min_distance, dist)
        
        return min_distance
    
    def _in_bounds(self, position: np.ndarray) -> bool:
        """Check if position is within environment bounds"""
        x, y = position[0], position[1]
        x_min, x_max, y_min, y_max = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    def _collides_with_obstacle(self, position: np.ndarray, obstacle: Obstacle) -> bool:
        """Check collision with specific obstacle"""
        if obstacle.type == ObstacleType.CIRCLE:
            return self._collides_circle(position, obstacle)
        elif obstacle.type == ObstacleType.RECTANGLE:
            return self._collides_rectangle(position, obstacle)
        elif obstacle.type == ObstacleType.POLYGON:
            return self._collides_polygon(position, obstacle)
        return False
    
    def _collides_circle(self, position: np.ndarray, obstacle: CircleObstacle) -> bool:
        """Circle-circle collision (robot modeled as circle)"""
        distance = np.linalg.norm(position[:2] - obstacle.position)
        return distance < (self.robot_radius + obstacle.radius)
    
    def _collides_rectangle(self, position: np.ndarray, obstacle: RectangleObstacle) -> bool:
        """Circle-rectangle collision"""
        # Transform position to rectangle's local frame
        rel_pos = position[:2] - obstacle.position
        
        if obstacle.angle != 0:
            # Rotate position to axis-aligned frame
            cos_a, sin_a = np.cos(-obstacle.angle), np.sin(-obstacle.angle)
            rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rel_pos = rotation @ rel_pos
        
        # Find closest point on rectangle to circle center
        half_w, half_h = obstacle.width / 2, obstacle.height / 2
        closest_x = np.clip(rel_pos[0], -half_w, half_w)
        closest_y = np.clip(rel_pos[1], -half_h, half_h)
        closest_point = np.array([closest_x, closest_y])
        
        # Check distance from circle center to closest point
        distance = np.linalg.norm(rel_pos - closest_point)
        return distance < self.robot_radius
    
    def _collides_polygon(self, position: np.ndarray, obstacle: PolygonObstacle) -> bool:
        """Circle-polygon collision using separating axis theorem"""
        # Check if circle center is inside polygon
        if self._point_in_polygon(position[:2], obstacle.vertices):
            return True
        
        # Check distance to each edge
        vertices = obstacle.vertices
        n = len(vertices)
        
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            
            dist = self._point_to_segment_distance(position[:2], v1, v2)
            if dist < self.robot_radius:
                return True
        
        return False
    
    def _distance_to_obstacle(self, position: np.ndarray, obstacle: Obstacle) -> float:
        """Signed distance to obstacle (negative inside, positive outside)"""
        if obstacle.type == ObstacleType.CIRCLE:
            dist = np.linalg.norm(position[:2] - obstacle.position)
            return dist - obstacle.radius - self.robot_radius
        elif obstacle.type == ObstacleType.RECTANGLE:
            # Simplified: distance to center minus inflated half-diagonal
            dist = np.linalg.norm(position[:2] - obstacle.position)
            half_diag = np.sqrt(obstacle.width**2 + obstacle.height**2) / 2
            return dist - half_diag - self.robot_radius
        elif obstacle.type == ObstacleType.POLYGON:
            # Distance to closest edge
            min_dist = float('inf')
            vertices = obstacle.vertices
            for i in range(len(vertices)):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % len(vertices)]
                dist = self._point_to_segment_distance(position[:2], v1, v2)
                min_dist = min(min_dist, dist)
            return min_dist - self.robot_radius
        
        return float('inf')
    
    def _penetration_depth(self, position: np.ndarray, obstacle: Obstacle) -> float:
        """How deep the robot penetrates the obstacle"""
        return max(0, -self._distance_to_obstacle(position, obstacle))
    
    @staticmethod
    def _point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
        """Ray casting algorithm for point-in-polygon test"""
        x, y = point
        n = len(vertices)
        inside = False
        
        p1x, p1y = vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    @staticmethod
    def _point_to_segment_distance(point: np.ndarray, v1: np.ndarray, 
                                   v2: np.ndarray) -> float:
        """Distance from point to line segment"""
        segment = v2 - v1
        segment_len_sq = np.dot(segment, segment)
        
        if segment_len_sq == 0:
            return np.linalg.norm(point - v1)
        
        # Project point onto line segment
        t = max(0, min(1, np.dot(point - v1, segment) / segment_len_sq))
        projection = v1 + t * segment
        
        return np.linalg.norm(point - projection)
    
    def visualize(self, ax=None, show_bounds: bool = True):
        """
        Visualize the environment (requires matplotlib).
        
        Args:
            ax: Matplotlib axis (creates new if None)
            show_bounds: Whether to show environment bounds
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle, Polygon
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw bounds
        if show_bounds:
            x_min, x_max, y_min, y_max = self.bounds
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  fill=False, edgecolor='black', linewidth=2))
        
        # Draw obstacles
        for obstacle in self.obstacles:
            if obstacle.type == ObstacleType.CIRCLE:
                circle = Circle(obstacle.position, obstacle.radius, 
                              color='red', alpha=0.5)
                ax.add_patch(circle)
                
            elif obstacle.type == ObstacleType.RECTANGLE:
                rect = Rectangle(
                    obstacle.position - np.array([obstacle.width/2, obstacle.height/2]),
                    obstacle.width, obstacle.height,
                    angle=np.degrees(obstacle.angle),
                    color='blue', alpha=0.5
                )
                ax.add_patch(rect)
                
            elif obstacle.type == ObstacleType.POLYGON:
                poly = Polygon(obstacle.vertices, color='green', alpha=0.5)
                ax.add_patch(poly)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Static Environment')
        
        return ax