#!/usr/bin/env python3
"""Test polygon vertices flattening for GPU kernel usage"""

import numpy as np
from environments.staticEnv import StaticEnvironment

def test_polygon_flattening():
    env = StaticEnvironment()
    
    # Add multiple polygons with different vertex counts
    poly1 = np.array([[0.0, 0.0], [1.0, 0.5], [0.5, 1.5]], dtype=np.float32)  # 3 vertices
    poly2 = np.array([[5.0, 5.0], [6.0, 5.0], [6.5, 6.0], [5.5, 6.5]], dtype=np.float32)  # 4 vertices
    poly3 = np.array([[10.0, 10.0], [11.0, 10.0], [11.5, 10.5], [11.0, 11.0], [10.5, 10.5]], dtype=np.float32)  # 5 vertices
    
    env.add_polygon_obstacle(poly1)
    env.add_polygon_obstacle(poly2)
    env.add_polygon_obstacle(poly3)
    
    obs_data = env.get_obstacle_data()
    
    print("=" * 70)
    print("POLYGON FLATTENING TEST")
    print("=" * 70)
    
    poly_data = obs_data['polygons']
    
    print(f"\nNumber of polygons: {poly_data['count']}")
    print(f"Num vertices per polygon: {poly_data['num_vertices']}")
    print(f"Poly starts: {poly_data['starts']}")
    print(f"Poly lengths: {poly_data['lengths']}")
    
    print(f"\nFlattened vertices shape: {poly_data['vertices_flat'].shape}")
    print(f"Flattened vertices:\n{poly_data['vertices_flat']}")
    
    # Verify correctness: reconstruct each polygon from flattened array
    print("\n" + "-" * 70)
    print("VERIFICATION: Reconstructing polygons from flattened data")
    print("-" * 70)
    
    for i in range(poly_data['count']):
        start = poly_data['starts'][i]
        length = poly_data['lengths'][i]
        reconstructed = poly_data['vertices_flat'][start:start+length]
        
        print(f"\nPolygon {i}:")
        print(f"  Start index: {start}")
        print(f"  Length: {length}")
        print(f"  Original vertices:\n{poly_data['vertices_list'][i]}")
        print(f"  Reconstructed vertices:\n{reconstructed}")
        
        # Check if they match
        if np.allclose(poly_data['vertices_list'][i], reconstructed):
            print(f"  ✓ Match!")
        else:
            print(f"  ✗ Mismatch!")
    
    print("\n" + "=" * 70)
    print("✅ Test completed!")
    print("=" * 70)
    
    # Example of how to use in GPU kernel
    print("\nExample GPU kernel usage:")
    print("""
    @cuda.jit
    def some_kernel(..., poly_vertices, poly_starts, poly_lengths, ...):
        p = 0  # polygon index
        start = poly_starts[p]
        length = poly_lengths[p]
        vertices = poly_vertices[start:start+length]
        # Now process polygon p using vertices
    """)

if __name__ == "__main__":
    test_polygon_flattening()
