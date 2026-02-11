import numpy as np
from numba import cuda, jit
import math
from functools import wraps

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

# =========================================================================
# Host (CPU) implementations of dynamics functions for testing and comparison
# =========================================================================
@dynamics_metadata(state_dim=3, control_dim=2, params_dim=1, description="Differential Drive Dynamics (host)")
def differential_drive_host(x, u, dt, params, x_next=None):
    px, py, theta = x[0], x[1], x[2]
    vr, vl = u[0], u[1]
    L = params[0]

    v = (vr + vl) / 2.0
    omega = (vr - vl) / L

    px_next = px + v * math.cos(theta) * dt
    py_next = py + v * math.sin(theta) * dt
    theta_next = theta + omega * dt

    return np.array([px_next, py_next, theta_next], dtype=np.float32)


@dynamics_metadata(state_dim=4, control_dim=2, params_dim=1, description="Bicycle Dynamics (host)")
def bicycle_dynamics_host(x, u, dt, params, x_next=None):
    px, py, theta, v = x[0], x[1], x[2], x[3]
    a, delta = u[0], u[1]
    L = params[0]

    px_next = px + v * math.cos(theta) * dt
    py_next = py + v * math.sin(theta) * dt
    theta_next = theta + (v / L) * math.tan(delta) * dt
    v_next = v + a * dt

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