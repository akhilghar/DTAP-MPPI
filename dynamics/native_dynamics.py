import numpy as np
from numba import cuda, jit
import math
from functools import wraps
from typing import Callable

# Metadata for the dynamics functions, to be used for dynamic kernel generation
def dynamics_metadata(state_dim, control_dim, params_dim, description=""):
    def decorator(func):
        func.metadata = {
            'state_dim': state_dim,
            'control_dim': control_dim,
            'params_dim': params_dim,
            'description': description,
            'name': func.__name__
        }
        return func
    return decorator 

def sample_terrain_slope(px, py, terrain, terrain_info):
    # Unpack terrain info
    xmin, ymin, dx, dy = terrain_info
    nx = terrain.shape[0]
    ny = terrain.shape[1]

    # Compute grid indices
    gx = (px - xmin) / dx
    gy = (py - ymin) / dy

    ix = int(gx)
    iy = int(gy)

    # Clamp indices to be within bounds
    ix = max(0, min(nx - 2, ix))
    iy = max(0, min(ny - 2, iy))

    # Fractional offsets for bilinear interpolation
    fx = gx - ix
    fy = gy - iy

    # Bilinear interpolation of slope values
    slope_x = (
        terrain[ix, iy] * (1 - fx) * (1 - fy) +
        terrain[ix + 1, iy] * fx * (1 - fy) +
        terrain[ix, iy + 1] * (1 - fx) * fy +
        terrain[ix + 1, iy + 1] * fx * fy
    )
    slope_y = (
        terrain[ix, iy] * (1 - fx) * (1 - fy) +
        terrain[ix + 1, iy] * fx * (1 - fy) +
        terrain[ix, iy + 1] * (1 - fx) * fy +
        terrain[ix + 1, iy + 1] * fx * fy
    )

    return slope_x, slope_y

# =========================================================================
# Host (CPU) implementations of dynamics functions for testing and comparison
# =========================================================================
@dynamics_metadata(state_dim=5, control_dim=2, params_dim=2, description="Differential Drive Dynamics (host)")
def differential_drive_host(x, u, dt, params, terrain, terrain_info):
    # Unpack state and control
    px, py, theta, pitch, roll = x
    vr, vl = u
    L, Cd = params  # Wheelbase, drag coefficient

    # Terrain Slope Sampling
    eps = 1e-3
    slopes = sample_terrain_slope(px, py, terrain, terrain_info)  # Should return (slope_x, slope_y)
    slope_x, slope_y = slopes

    slope = slope_x * math.cos(theta) + slope_y * math.sin(theta)
    slope_angle = math.atan(slope)

    lateral_slope = -slope_x * math.sin(theta) + slope_y * math.cos(theta)
    lateral_slope_angle = math.atan(lateral_slope)

    # Gravity compensation
    g = 9.81
    gravity_comp = g * math.sin(slope_angle)

    # Compute wheel velocities with drag compensation
    terrain_resistance = Cd * abs(slope)
    v_eff_scale = max(0.0, 1.0 - (terrain_resistance + gravity_comp)*dt)
    vr_eff = vr * v_eff_scale
    vl_eff = vl * v_eff_scale

    v = (vr_eff + vl_eff) / 2.0
    omega = (vr_eff - vl_eff) / L

    # Compute next state using differential drive kinematics
    px_next = px + v * math.cos(theta) * dt
    py_next = py + v * math.sin(theta) * dt
    theta_next = theta + omega * dt

    # Add pitch and roll dynamics based on terrain slope
    alpha = 0.3
    pitch_next = pitch + alpha * (slope_angle - pitch)
    roll_next = roll + alpha * (lateral_slope_angle - roll)
    
    return np.array([px_next, py_next, theta_next, pitch_next, roll_next], dtype=np.float32)


@dynamics_metadata(state_dim=4, control_dim=2, params_dim=1, description="Bicycle Dynamics (host)")
def bicycle_dynamics_host(x, u, dt, params, x_next=None):
    px, py, theta, v = x[0], x[1], x[2], x[3]
    a, delta = u[0], u[1]
    L = params[0]

    px_next = px + v * math.cos(theta) * dt
    py_next = py + v * math.sin(theta) * dt
    theta_next = theta + (v / L) * math.tan(delta) * dt
    v_next = max(0.0, v + a * dt)

    return np.array([px_next, py_next, theta_next, v_next], dtype=np.float32)


@dynamics_metadata(state_dim=4, control_dim=2, params_dim=0, description="Double Integrator Dynamics (host)")
def double_integrator_host(x, u, dt, params=None, x_next=None):
    px, py, vx, vy = x[0], x[1], x[2], x[3]
    ax, ay = u[0], u[1]

    px_next = px + vx * dt + 0.5 * ax * dt * dt
    py_next = py + vy * dt + 0.5 * ay * dt * dt
    vx_next = vx + ax * dt
    vy_next = vy + ay * dt

    return np.array([px_next, py_next, vx_next, vy_next], dtype=np.float32)


@dynamics_metadata(state_dim=4, control_dim=2, params_dim=1, description="Ackermann Steering Dynamics (host)")
def ackermann_dynamics_host(x, u, dt, params, x_next=None):
    px, py, theta, v = x[0], x[1], x[2], x[3]
    a, delta = u[0], u[1]
    L = params[0]

    beta = math.atan((L / 2.0) * math.tan(delta) / L)

    px_next = px + v * math.cos(theta + beta) * dt
    py_next = py + v * math.sin(theta + beta) * dt
    theta_next = theta + (v / L) * math.sin(beta) * dt
    v_next = v + a * dt

    return np.array([px_next, py_next, theta_next, v_next], dtype=np.float32)