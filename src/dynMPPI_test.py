# test_dynamic_env.py

import numpy as np
import matplotlib.pyplot as plt
import time
from controllers.mppi_dynObs import MPPIDynObs, MPPIConfig
from dynamics.models import DYNAMICS_REGISTRY
from environments.dynamicEnv import DynamicEnv, Obstacle, ObstacleMode

# ============================================================================
# Setup Environment
# ============================================================================
corridor_width = 4.0
env = DynamicEnv(bounds=(-corridor_width, corridor_width, -4, 24), robot_radius=0.3)

# Add moving circular obstacles
rng = np.random.default_rng(seed=42)
for i in range(0,8):
    env.add_obstacle(
        Obstacle(position=[np.random.randint(-corridor_width+1, corridor_width-1), np.random.randint(1.0, 22.0)], 
                 radius=0.3+0.4*np.random.rand(),
                 velocity=[2.0*np.random.rand()-1.0, 2.0*np.random.rand()-1.0],
                 mode=ObstacleMode.AVOIDANT)
    )

# ============================================================================
# Configure MPPI
# ============================================================================

# Define function used, reference this function exclusively
model_name = "differential_drive_noslope"  # "differential_drive", "ackermann", "bicycle"
model = DYNAMICS_REGISTRY[model_name]

model_md = model.metadata
state_dim = model_md["state_dim"]
control_dim = model_md["control_dim"]

print("State Dimensions: ", state_dim)

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
    umin_mod = np.array([-3.0, -3.0])
    umax_mod = np.array([3.0, 3.0])
    noise_mod = np.array([0.5, 0.5])
    ctrl_label_1 = "Left Wheel Velocity"
    ctrl_label_2 = "Right Wheel Velocity"
    x0 = np.array([0.0, 0.0, 0.0])
    x_goal = np.array([0.0, 20.0, 0.0])

config = MPPIConfig(
    num_samples=20000,
    horizon=40,
    dt=0.05,
    lambda_=30.0, # increase temperature for smoother trajectory

    Q=Q_mod,
    Qf=Qf_mod,
    R=R_mod,

    Q_obs=190.0,
    d_safe=env.robot_radius + 0.1,

    dynamics_params=np.array([2*env.robot_radius]),

    u_min=umin_mod,
    u_max=umax_mod,

    noise_sigma=noise_mod,
)

# print(config)

mppi = MPPIDynObs(config, model.gpu, environment=env, dem=None)

# ============================================================================
# Simulation
# ============================================================================

trajectory = [x0.copy()]
controls = []
cov_log = []

obstacle_history = []
rollout_snapshots = {}  # step -> (expected_traj, sample_trajs), sampled every 20 steps

x = x0.copy()
num_steps = 500
num_safe = 0

print("Running Dynamic MPPI Simulation...")
sim_start_time = time.time()
for step in range(num_steps):

    # --- Step environment first (obstacles move) ---
    env.move_obstacles(config.dt, robot_pos=x[:2])
    obstacle_history.append(
        np.array([obs.position.copy() for obs in env.obstacles])
    )

    # --- Get MPPI control ---
    start = time.time()
    u, is_safe = mppi.get_control(x, x_goal)
    if is_safe:
        num_safe += 1

    cov = mppi.get_covariance()
    end = time.time()

    # --- Apply dynamics ---
    x = model.cpu(x, u, config.dt, config.dynamics_params)

    if env.check_for_collision(x[:2]):
        print("FATAL: Robot has been killed by the environment. Terminating simulation.")
        break

    trajectory.append(x.copy())
    controls.append(u.copy())
    cov_log.append(cov)

    if state_dim == 4:
        vel = x[3]
    else:
        vel = 0.5*u[0] + 0.5*u[1]

    # --- Goal check ---
    if (np.linalg.norm(x[:2] - x_goal[:2]) < env.robot_radius) & (vel < 0.01):
        print(f"Reached goal at step {step}")
        break

    if step % 20 == 0:
        rollout_snapshots[step] = mppi.get_rollout_snapshot(n=5)
        print(f"Step {step}: pos=({x[0]:.2f},{x[1]:.2f}), "
              f"covariance={cov}, "
              f"safe={is_safe}, "
              f"time per step={end-start:.3f}s")

trajectory = np.array(trajectory)
controls = np.array(controls)
obstacle_history = np.array(obstacle_history)
sim_end_time = time.time()

print(f"Simulation complete: {len(trajectory)} steps")
print(f"Safe Trajectory Rate: {num_safe/len(trajectory):.2f}")
print(f"Total Simulation Time: {sim_end_time - sim_start_time:.2f} seconds")
cov_log = np.array(cov_log)

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

# Rollout visualization: sample trajectories + selected (weighted-mean) trajectory
N_DISPLAY_ROLLOUTS = 5
sample_rollout_lines = [
    ax.plot([], [], color='orange', alpha=0.25, linewidth=0.8, zorder=1)[0]
    for _ in range(N_DISPLAY_ROLLOUTS)
]
selected_rollout_line, = ax.plot([], [], color='lime', alpha=0.85, linewidth=1.5, zorder=2)

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

    # Update rollout lines using the most recent snapshot (every 20 frames)
    snapshot_step = (frame // 20) * 20
    if snapshot_step in rollout_snapshots:
        exp_traj, sample_trajs = rollout_snapshots[snapshot_step]
        if exp_traj is not None:
            selected_rollout_line.set_data(exp_traj[:, 0], exp_traj[:, 1])
        for j, line in enumerate(sample_rollout_lines):
            line.set_data(sample_trajs[j, :, 0], sample_trajs[j, :, 1])

    return ([robot_patch, traj_line, heading_line, selected_rollout_line]
            + sample_rollout_lines + obstacle_patches)

ani = animation.FuncAnimation(
    fig,
    update,
    frames=min(len(trajectory), len(obstacle_history)),
    interval=config.dt * 1000,  # milliseconds
    blit=True
)
t_fin = sim_end_time - sim_start_time
plt.show()
ani.save(f"./media/GIFs/mppi_animation_{model_name}_{t_fin:.2f}_prob.gif", writer="pillow", fps=1/config.dt)
print("Saved animated GIF of Robot.")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

time_vec = np.arange(len(controls)) * config.dt

# Plot 1: Covariance Evolution
ax1 = axes[0]
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Covariance")
ax1.plot(time_vec, cov_log[:, 0], label='Control Input 1 Covariance')
ax1.plot(time_vec, cov_log[:, 1], label='Control Input 2 Covariance')
ax1.legend()
ax1.grid(True)
ax1.set_title("Covariance Evolution")

# Plot 2: Robot Velocity
ax2 = axes[1]
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Velocity (m/s)")
if state_dim == 4:
    ax2.plot(time_vec, trajectory[:-1, 3], label='Velocity')
else:
    v_avg = 0.5*controls[:,0] + 0.5*controls[:,1]
    ax2.plot(time_vec, v_avg, label='Velocity')
ax2.set_title("Robot Velocity")
ax2.grid(True)

# Plot 3: Steering Angle
ax3 = axes[2]
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Steering Angle (rad)")
if state_dim == 4:
    ax3.plot(time_vec, controls[:, 1], label='Steering Angle')
    ax3.set_title("Steering Angle")
    ax3.grid(True)
else:
    ax3.plot(time_vec, trajectory[:-1, 2], label='Steering Angle')
    ax3.set_title("Steering Angle")
    ax3.grid(True)

plt.tight_layout()
filename = f'./media/Visualizations/mppi_result_{model_name}_{t_fin:.2f}_dynProb.png'
plt.savefig(filename, dpi=150)
# plt.show()

print("Visualization Saved.")
mppi.free_gpu_buffers()