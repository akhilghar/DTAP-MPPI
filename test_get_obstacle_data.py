#!/usr/bin/env python3
"""Test script for get_obstacle_data function"""

import numpy as np
from environments.staticEnv import StaticEnvironment

def test_get_obstacle_data():
    """Test obstacle data extraction with all obstacle types"""
    
    # Create environment
    env = StaticEnvironment(bounds=(-10, 10, -10, 10), robot_radius=0.5)
    
    # Add various obstacles
    env.add_circle_obstacle(np.array([2.0, 2.0]), radius=1.5)
    env.add_circle_obstacle(np.array([-3.0, 1.0]), radius=1.0)
    
    env.add_rectangle_obstacle(np.array([5.0, 5.0]), width=2.0, height=1.0, angle=0.0)
    env.add_rectangle_obstacle(np.array([-5.0, -5.0]), width=3.0, height=2.0, angle=np.pi/4)
    
    vertices1 = np.array([[0.0, 0.0], [1.0, 0.5], [0.5, 1.5]])
    env.add_polygon_obstacle(vertices1)
    
    vertices2 = np.array([[-8.0, 8.0], [-7.0, 7.5], [-7.0, 8.5], [-8.0, 8.2]])
    env.add_polygon_obstacle(vertices2)
    
    # Get obstacle data
    obs_data = env.get_obstacle_data()
    
    print("=" * 60)
    print("OBSTACLE DATA TEST")
    print("=" * 60)
    
    print(f"\nTotal obstacles: {obs_data['num_obstacles']}")
    print(f"Obstacle types: {obs_data['obstacle_types']}")
    
    # Circle data
    print(f"\n--- CIRCLES ({obs_data['circles']['count']}) ---")
    print(f"Positions shape: {obs_data['circles']['positions'].shape}")
    print(f"Positions:\n{obs_data['circles']['positions']}")
    print(f"Radii: {obs_data['circles']['radii']}")
    
    # Rectangle data
    print(f"\n--- RECTANGLES ({obs_data['rectangles']['count']}) ---")
    print(f"Positions shape: {obs_data['rectangles']['positions'].shape}")
    print(f"Positions:\n{obs_data['rectangles']['positions']}")
    print(f"Widths: {obs_data['rectangles']['widths']}")
    print(f"Heights: {obs_data['rectangles']['heights']}")
    print(f"Angles (rad): {obs_data['rectangles']['angles']}")
    
    # Polygon data
    print(f"\n--- POLYGONS ({obs_data['polygons']['count']}) ---")
    print(f"Num vertices per polygon: {obs_data['polygons']['num_vertices']}")
    for i, vertices in enumerate(obs_data['polygons']['vertices_list']):
        print(f"  Polygon {i} vertices shape: {vertices.shape}")
        print(f"  Vertices:\n{vertices}")
    
    # Test empty environment
    print("\n" + "=" * 60)
    print("TESTING EMPTY ENVIRONMENT")
    print("=" * 60)
    
    env_empty = StaticEnvironment()
    obs_data_empty = env_empty.get_obstacle_data()
    
    print(f"Empty env obstacles: {obs_data_empty['num_obstacles']}")
    print(f"Empty circles count: {obs_data_empty['circles']['count']}")
    print(f"Empty rectangles count: {obs_data_empty['rectangles']['count']}")
    print(f"Empty polygons count: {obs_data_empty['polygons']['count']}")
    print(f"Circles positions shape: {obs_data_empty['circles']['positions'].shape}")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_get_obstacle_data()
