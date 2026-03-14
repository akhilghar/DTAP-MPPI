# main.py
import numpy as np
import matplotlib.pyplot as plt
from controllers.mppi_baseline import MPPIBaseline, MPPIConfig
from dynamics.models import DYNAMICS_REGISTRY
from environments.staticEnv import StaticEnvironment
# from environments.dynamicEnv_probabilistic import ProbabilisticEnv, Obstacle, ObstacleMode
from environments.dynamicEnv_deterministic import DeterministicEnv, Obstacle
import time
import os

# ============================================================================
# Setup
# ============================================================================

# Create environment with obstacles
corridor_width = 4.0
env = DeterministicEnv(bounds=(-corridor_width, corridor_width, -4, 24), robot_radius=0.2)

# Add moving circular obstacles
rng = np.random.default_rng(seed=42)
for i in range(0,8):
    env.add_obstacle(
        Obstacle(position=[np.random.randint(-corridor_width+1, corridor_width-1), np.random.randint(1.0, 22.0)], 
                 radius=0.3+0.4*np.random.rand(),
                 velocity=[2.0*np.random.rand()-1.0, 2.0*np.random.rand()-1.0])
    )

# Configure MPPI

model_name = "differential_drive"  # "differential_drive", "ackermann", "bicycle"
model = DYNAMICS_REGISTRY[model_name]

model_md = model.metadata
state_dim = model_md["state_dim"]
control_dim = model_md["control_dim"]

max_deg = 75.0
if state_dim == 4:
    Q_mod=np.diag([10.0, 10.0, 2.0, 10.0])
    Qf_mod=np.diag([50.0, 50.0, 10.0, 50.0])
    R_mod = np.diag([0.1, 0.1])
    umin_mod = np.array([-2.0, -max_deg*np.pi/180])
    umax_mod = np.array([2.0, max_deg*np.pi/180])
    noise_mod = np.array([0.55, 0.15])
    ctrl_label_1 = "Acceleration"
    ctrl_label_2 = "Steering Angle"
    x0 = np.array([0.0, 0.0, np.pi/2, 0.0])
    x_goal = np.array([0.0, 20.0, np.pi/2, 0.0])

else:
    Q_mod=np.diag([7.0, 7.0, 1.5])
    Qf_mod=np.diag([40.0, 40.0, 5.0])
    R_mod = np.eye(control_dim) * 5.0
    umin_mod = np.array([-4.0, -4.0])
    umax_mod = np.array([4.0, 4.0])
    noise_mod = np.array([0.5, 0.5])
    ctrl_label_1 = "Left Wheel Velocity"
    ctrl_label_2 = "Right Wheel Velocity"
    x0 = np.array([0.0, 0.0, 0.0])
    x_goal = np.array([0.0, 20.0, 0.0])

max_deg = 72.0 # maximum steering angle in degrees
config = MPPIConfig(
    num_samples=20000,
    horizon=50,
    dt=0.05,
    lambda_=15.0, # increase temperature for smoother trajectory

    Q=Q_mod,
    Qf=Qf_mod,
    R=R_mod,

    Q_obs=250.0,
    d_safe=env.robot_radius + 0.1,

    dynamics_params=np.array([1.0]),

    u_min=umin_mod,
    u_max=umax_mod,

    noise_sigma=noise_mod,
)

# Create MPPI controller
mppi = MPPIBaseline(config, model.gpu, environment=env)

# ============================================================================
# Simulate
# ============================================================================

trajectory = [x0.copy()]
controls = []
num_steps = 500
num_safe = 0

x = x0.copy()
obstacle_history = []
goal_reached = 0
sim_start = time.time()

print("Running MPPI simulation...")
for step in range(num_steps):
    # Step environment (update obstacles) and record for visualization
    env.step(config.dt)
    obstacle_history.append(np.array([obs.position.copy() for obs in env.obstacles]))

    # Get control (returns (u_opt, is_safe))
    start = time.time()
    u, is_safe = mppi.get_control(x, x_goal)
    if is_safe:
        num_safe += 1
    
    # Apply control (using host dynamics implementation)
    x_next = model.cpu(x, u, config.dt, config.dynamics_params)

    # Check collision against the new state
    if env.check_for_collision(x_next[:2]):
        print("FATAL: Robot has been killed by the environment. Terminating simulation.")
        break

    # Commit new state and record
    trajectory.append(x_next.copy())
    controls.append(u.copy())
    x = x_next

    if state_dim == 4:
        vel = x[3]
    else:
        vel = 0.5*u[0] + 0.5*u[1]

    # --- Goal check ---
    if (np.linalg.norm(x[:2] - x_goal[:2]) < env.robot_radius) & (vel < 0.01):
        print(f"Reached goal at step {step}")
        break
    
    if step % 20 == 0:
        end = time.time()
        print(f"Step {step}: pos=({x[0]:.2f}, {x[1]:.2f}), safe={is_safe}, time={end-start:.3f}s")

trajectory = np.array(trajectory)
controls = np.array(controls)
obstacle_history = np.array(obstacle_history)
sim_end = time.time()

print(f"Simulation complete: {len(trajectory)} steps")
print(f"Safe Trajectory Rate: {num_safe/len(trajectory):.2f}")
print(f"Total Simulation Time: {sim_end-sim_start:.2f} seconds")

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
    # Return the artists that have been modified (required for blitting)
    return [robot_patch, heading_line] + obstacle_patches

ani = animation.FuncAnimation(
    fig,
    update,
    frames=max(1, min(len(trajectory), len(obstacle_history))),
    interval=config.dt * 1000,  # milliseconds
    blit=True,
)

# Ensure output directories exist
gif_dir = os.path.join('media', 'GIFs')
vis_dir = os.path.join('media', 'Visualizations')
os.makedirs(gif_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

t_fin = sim_end - sim_start
plt.show()
gif_path = os.path.join(gif_dir, f'mppi_animation_{model_name}_{t_fin:.2f}_staticDet.gif')
try:
    ani.save(gif_path, writer='pillow', fps=max(1, int(1.0/config.dt)))
    print("Saved animated GIF of Robot to", gif_path)
except Exception as e:
    print("Could not save GIF (pillow writer may be missing):", e)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Trajectory
ax1 = axes[0, 0]
ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
ax1.plot(x0[0], x0[1], 'go', markersize=12, label='Start')
ax1.plot(x_goal[0], x_goal[1], 'r*', markersize=15, label='Goal')
ax1.legend()
ax1.set_title('MPPI Trajectory with Obstacle Avoidance')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.set_xlim(env.bounds[0]-1, env.bounds[1]+1)

# Plot 2: Controls
ax2 = axes[0, 1]
time = np.arange(len(controls)) * config.dt
ax2.plot(time, controls[:, 0], label='Acceleration', linewidth=2)
ax2.plot(time, controls[:, 1], label='Steering', linewidth=2)
ax2.axhline(config.u_max[0], color='r', linestyle='--', alpha=0.5, label='Limits')
ax2.axhline(config.u_min[0], color='r', linestyle='--', alpha=0.5)
ax2.axhline(config.u_max[1], color='orange', linestyle='--', alpha=0.5)
ax2.axhline(config.u_min[1], color='orange', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Control')
ax2.legend()
ax2.grid(True)
ax2.set_title('Control Inputs')

# Plot 3: State evolution
ax3 = axes[1, 0]
ax3.plot(time, trajectory[:-1, 2], label='Heading θ')
if state_dim == 4:
        ax3.plot(time, trajectory[:-1, 3], label='Velocity v')
else:
    v_avg = 0.5*controls[:,0] + 0.5*controls[:,1]
    ax3.plot(time, v_avg, label='Velocity')

ax3.set_xlabel('Time (s)')
ax3.set_ylabel('State')
ax3.legend()
ax3.grid(True)
ax3.set_title('State Evolution')

# Plot 4: Distance to goal
ax4 = axes[1, 1]
dist_to_goal = np.linalg.norm(trajectory[:, :2] - x_goal[:2], axis=1)
ax4.plot(time, dist_to_goal[:-1], linewidth=2)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Distance to Goal (m)')
ax4.grid(True)
ax4.set_title('Convergence to Goal')

plt.tight_layout()
filename = f'media/Visualizations/mppi_result_{model_name}_{t_fin:.2f}_staticDet.png'
plt.savefig(filename, dpi=150)
# plt.show()

print("Visualization saved to", filename)
mppi.free_gpu_buffers()