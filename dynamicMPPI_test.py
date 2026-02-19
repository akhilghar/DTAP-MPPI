# test_dynamic_env.py

import numpy as np
import math
import matplotlib.pyplot as plt
import time
from controllers.mppi_dynObs import MPPIDynObs, MPPIConfig
from dynamics.models import DYNAMICS_REGISTRY
from environments.dynamicEnv_deterministic import DeterministicEnv, Obstacle

# ============================================================================
# Setup Environment
# ============================================================================

env = DeterministicEnv(bounds=(-2, 12, -2, 12), robot_radius=0.25)

# Add moving circular obstacles
env.add_obstacle(
    Obstacle(position=[5.0, 5.0], radius=1.0, velocity=[-1.34, 0])
)

env.add_obstacle(
    Obstacle(position=[3.0, 0.0], radius=0.8, velocity=[0, -1.34])
)

env.add_obstacle(
    Obstacle(position=[6.0, 8.0], radius=0.6, velocity=[1.0, -0.9])
)

env.add_obstacle(
    Obstacle(position=[3.0, 1.0], radius=0.4, velocity=[-1.0, 0.9])
)

env.add_obstacle(
    Obstacle(position=[8.0, 2.0], radius=1.6, velocity=[0.5, 0.3])
)

# env.add_obstacle(
#    Obstacle(position=[0.0, 8.0], radius=0.5, velocity=[-1.0, -0.2])
#)

# ============================================================================
# Configure MPPI
# ============================================================================

# Define function used, reference this function exclusively
model_name = "bicycle"
model = DYNAMICS_REGISTRY[model_name]

model_md = model.metadata
state_dim = model_md["state_dim"]
control_dim = model_md["control_dim"]

print("State Dimensions: ", state_dim)

max_deg = 60.0
if state_dim == 4:
    Q_mod=np.diag([10.0, 10.0, 2.0, 2.0])
    Qf_mod=np.diag([50.0, 50.0, 5.0, 5.0])
    umin_mod = np.array([-3.0, -max_deg*np.pi/180])
    umax_mod = np.array([3.0, max_deg*np.pi/180])
    noise_mod = np.array([0.6, 0.25])
    ctrl_label_1 = "Acceleration"
    ctrl_label_2 = "Steering Angle"
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    x_goal = np.array([10.0, 10.0, 0.0, 0.0])

else:
    Q_mod=np.diag([7.0, 7.0, 1.0])
    Qf_mod=np.diag([45.0, 45.0, 5.0])
    umin_mod = np.array([-0.5, -0.5])
    umax_mod = np.array([4.0, 4.0])
    noise_mod = np.array([0.5, 0.5])
    ctrl_label_1 = "Left Wheel Velocity"
    ctrl_label_2 = "Right Wheel Velocity"
    x0 = np.array([0.0, 0.0, 0.0])
    x_goal = np.array([10.0, 10.0, 0.0])


config = MPPIConfig(
    num_samples=10000,
    horizon=40,
    dt=0.05,
    lambda_=10.0, # increase temperature for smoother trajectory

    Q=Q_mod,
    Qf=Qf_mod,
    R=np.diag([0.1, 0.1]),

    Q_obs=150.0,
    d_safe=env.robot_radius + 0.1,

    dynamics_params=np.array([1.0]),

    u_min=umin_mod,
    u_max=umax_mod,

    noise_sigma=noise_mod,
)

# print(config)

mppi = MPPIDynObs(config, model.gpu, environment=env)

# ============================================================================
# Simulation
# ============================================================================

trajectory = [x0.copy()]
controls = []

obstacle_history = []

x = x0.copy()
num_steps = 300

print("Running Dynamic MPPI Simulation...")

for step in range(num_steps):

    # --- Step environment first (obstacles move) ---
    env.step(config.dt)
    obstacle_history.append(
        np.array([obs.position.copy() for obs in env.obstacles])
    )

    # --- Get MPPI control ---
    start = time.time()
    u, is_safe = mppi.get_control(x, x_goal)
    end = time.time()

    # --- Apply dynamics ---
    x = model.cpu(x, u, config.dt, config.dynamics_params)

    if env.check_for_collision(x[:2]):
        print("FATAL: Robot has been killed by the environment. Terminating simulation.")
        break

    trajectory.append(x.copy())
    controls.append(u.copy())

    # --- Goal check ---
    if np.linalg.norm(x[:2] - x_goal[:2]) < 0.5:
        print(f"Reached goal at step {step}")
        break

    if step % 20 == 0:
        print(f"Step {step}: pos=({x[0]:.2f},{x[1]:.2f}), "
              f"safe={is_safe}, "
              f"time={end-start:.3f}s")

trajectory = np.array(trajectory)
controls = np.array(controls)
obstacle_history = np.array(obstacle_history)

print(f"Simulation complete: {len(trajectory)} steps")

# ============================================================================
# Visualization
# ============================================================================

import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')

xmin, xmax, ymin, ymax = env.bounds
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Draw boundary
ax.add_patch(Rectangle((xmin, ymin),
                       xmax - xmin,
                       ymax - ymin,
                       fill=False,
                       linewidth=2))

# Robot
robot_patch = Circle((trajectory[0, 0], trajectory[0, 1]),
                     env.robot_radius,
                     color='blue')
ax.add_patch(robot_patch)

heading_length = mppi.config.d_safe
heading_line, = ax.plot([], [], 'k-', linewidth=2)

# Obstacles
obstacle_patches = []
for i in range(obstacle_history.shape[1]):
    circle = Circle(obstacle_history[0, i],
                    env.obstacles[i].radius,
                    color='red')
    ax.add_patch(circle)
    obstacle_patches.append(circle)

# Trajectory trail
traj_line, = ax.plot([], [], 'b-', linewidth=2)

# Goal
ax.plot(x_goal[0], x_goal[1], 'r*', markersize=15)

def update(frame):
    # State Acquisition
    x = trajectory[frame, 0]
    y = trajectory[frame, 1]
    theta = trajectory[frame, 2]

    # Update robot
    robot_patch.center = (x,y)
    dx = heading_length*np.cos(theta)
    dy = heading_length*np.sin(theta)

    heading_line.set_data(
        [x,x+dx],
        [y,y+dy]
    )

    # Update obstacles
    for i, patch in enumerate(obstacle_patches):
        patch.center = obstacle_history[frame, i]

    # Update trajectory trail
    traj_line.set_data(trajectory[:frame+1, 0],
                       trajectory[:frame+1, 1])

    return [robot_patch, traj_line, heading_line] + obstacle_patches

ani = animation.FuncAnimation(
    fig,
    update,
    frames=min(len(trajectory), len(obstacle_history)),
    interval=config.dt * 1000,  # milliseconds
    blit=True
)

plt.show()
ani.save(f"media/mppi_animation_{model_name}_{x_goal[0]}_{x_goal[1]}_{x0[2]:.2f}_det.gif", writer="pillow", fps=1/config.dt)
print("Saved animated GIF of Robot.")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Trajectory
ax1 = axes[0, 0]
xmin, xmax, ymin, ymax = env.bounds
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)
ax1.set_aspect("equal")

# Draw obstacles at final state
for obs in env.obstacles:
    circle = plt.Circle(obs.position, obs.radius, color='red', alpha=0.5)
    ax1.add_patch(circle)

ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
ax1.plot(x0[0], x0[1], 'go', markersize=10)
ax1.plot(x_goal[0], x_goal[1], 'r*', markersize=15)

ax1.set_title("Dynamic MPPI Trajectory")
ax1.grid(True)

# Plot 2: Controls
ax2 = axes[0, 1]
time_vec = np.arange(len(controls)) * config.dt
ax2.plot(time_vec, controls[:, 0], label='Control Input 1')
ax2.plot(time_vec, controls[:, 1], label='Control Input 2')
ax2.legend()
ax2.grid(True)
ax2.set_title("Control Inputs")

# Plot 3: States
ax3 = axes[1, 0]
ax3.plot(time_vec, trajectory[:-1, 2], label='Heading')
if state_dim == 4:
    ax3.plot(time_vec, trajectory[:-1, 3], label='Velocity')
else:
    v_avg = 0.5*controls[:,0] + 0.5*controls[:,1]
    ax3.plot(time_vec, v_avg, label='Velocity')
ax3.legend()
ax3.grid(True)
ax3.set_title("State Evolution")

# Plot 4: Distance to Goal
ax4 = axes[1, 1]
dist_to_goal = np.linalg.norm(trajectory[:, :2] - x_goal[:2], axis=1)
ax4.plot(time_vec, dist_to_goal[:-1])
ax4.set_title("Distance to Goal")
ax4.grid(True)

plt.tight_layout()
filename = f'media/mppi_result_{model_name}_{x_goal[0]}_{x_goal[1]}_{x0[2]:.2f}_dynDet.png'
plt.savefig(filename, dpi=150)
plt.show()

print("Visualization Saved.")
mppi.free_gpu_buffers()