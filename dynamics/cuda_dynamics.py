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

@cuda.jit(device=True)
def true_terrain_slope_cuda(px, py, terrain, terrain_info):
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
# Example dynamics functions that can be used with the MPPI controller
# =========================================================================

@dynamics_metadata(state_dim=5, control_dim=2, params_dim=2, description="Differential Drive Dynamics")
@cuda.jit(device=True)
def differential_drive(x, u, dt, params, terrain, terrain_info, x_next):
    # Unpack state and control
    px, py, theta, pitch, roll = x
    vr, vl = u
    L, Cd = params  # Wheelbase, drag coefficient

    # Terrain Slope Sampling
    eps = 1e-3
    slopes = true_terrain_slope_cuda(px, py, terrain, terrain_info)  # Should return (slope_x, slope_y)
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
    
    x_next[0] = px_next
    x_next[1] = py_next
    x_next[2] = theta_next
    x_next[3] = pitch_next
    x_next[4] = roll_next

@dynamics_metadata(state_dim=4, control_dim=2, params_dim=1, description="Bicycle Dynamics")
@cuda.jit(device=True)
def bicycle_dynamics(x, u, dt, params, x_next):
    # Unpack state and control
    px, py, theta, v = x
    a, delta = u
    L = params[0]  # Wheelbase

    # Compute next state using bicycle model kinematics
    px_next = px + v * math.cos(theta) * dt
    py_next = py + v * math.sin(theta) * dt
    theta_next = theta + (v / L) * math.tan(delta) * dt
    v_next = max(0.0, v + a * dt)
    
    x_next[0] = px_next
    x_next[1] = py_next
    x_next[2] = theta_next
    x_next[3] = v_next

@dynamics_metadata(state_dim=4, control_dim=2, params_dim=0, description="Double Integrator Dynamics")
@cuda.jit(device=True)
def double_integrator(x, u, dt, params, x_next):
    # Unpack state and control
    px, py, vx, vy = x
    ax, ay = u

    # Compute next state using double integrator dynamics
    px_next = px + vx * dt + 0.5 * ax * dt * dt
    py_next = py + vy * dt + 0.5 * ay * dt * dt
    vx_next = vx + ax * dt
    vy_next = vy + ay * dt
    
    x_next[0] = px_next
    x_next[1] = py_next
    x_next[2] = vx_next
    x_next[3] = vy_next

@dynamics_metadata(state_dim=4, control_dim=2, params_dim=1, description="Ackermann Steering Dynamics")
@cuda.jit(device=True)
def ackermann_dynamics(x, u, dt, params, x_next):
    # Unpack state and control
    px, py, theta, v = x
    a, delta = u
    L = params[0]  # Wheelbase

    # Compute next state using Ackermann steering model kinematics
    beta = math.atan((L / 2.0) * math.tan(delta) / L)

    px_next = px + v * math.cos(theta+beta) * dt
    py_next = py + v * math.sin(theta+beta) * dt
    theta_next = theta + (v / L) * math.sin(beta) * dt
    v_next = v + a * dt
    
    x_next[0] = px_next
    x_next[1] = py_next
    x_next[2] = theta_next
    x_next[3] = v_next