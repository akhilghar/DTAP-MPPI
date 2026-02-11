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
# Example dynamics functions that can be used with the MPPI controller
# =========================================================================

@dynamics_metadata(state_dim=3, control_dim=2, params_dim=1, description="Differential Drive Dynamics")
@cuda.jit(device=True)
def differential_drive(x, u, dt, params, x_next):
    # Unpack state and control
    px, py, theta = x
    vr, vl = u
    L = params[0]  # Wheelbase

    v = (vr + vl) / 2.0
    omega = (vr - vl) / L

    # Compute next state using differential drive kinematics
    px_next = px + v * math.cos(theta) * dt
    py_next = py + v * math.sin(theta) * dt
    theta_next = theta + omega * dt
    
    x_next[0] = px_next
    x_next[1] = py_next
    x_next[2] = theta_next

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
    v_next = v + a * dt
    
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