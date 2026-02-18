# main.py
import numpy as np
import matplotlib.pyplot as plt
from controllers.mppi_baseline import MPPIBaseline, MPPIConfig
from dynamics.models import DYNAMICS_REGISTRY
from environments.staticEnv import StaticEnvironment
import time

# ============================================================================
# Setup
# ============================================================================

# Create environment with obstacles
env = StaticEnvironment(bounds=(-2, 12, -2, 12), robot_radius=0.25)
env.add_circle_obstacle(np.array([5.0, 5.0]), radius=1.0)
env.add_circle_obstacle(np.array([7.0, 3.0]), radius=0.8)
env.add_circle_obstacle(np.array([3.0, 8.0]), radius=0.6)
env.add_rectangle_obstacle(np.array([9.0, 1.0]), width=1.5, height=0.5, angle=np.pi/6)
env.add_polygon_obstacle(np.array([[2.0, 1.5], [4.0, 1.0], [3.0, 3.0], [2.0, 3.0]]))  # Square obstacle

# Configure MPPI

model_name = "differential_drive"
model = DYNAMICS_REGISTRY[model_name]

model_md = model.metadata
state_dim = model_md["state_dim"]
control_dim = model_md["control_dim"]

max_deg = 60.0
if state_dim == 4:
    Q_mod=np.diag([10.0, 10.0, 2.0, 2.0])
    Qf_mod=np.diag([50.0, 50.0, 5.0, 5.0])
    umin_mod = np.array([-3.0, -max_deg*np.pi/180])
    umax_mod = np.array([3.0, max_deg*np.pi/180])
    noise_mod = np.array([0.5, 0.2])
    ctrl_label_1 = "Acceleration"
    ctrl_label_2 = "Steering Angle"
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    x_goal = np.array([10.0, 10.0, 0.0, 0.0])

else:
    Q_mod=np.diag([10.0, 10.0, 2.0])
    Qf_mod=np.diag([50.0, 50.0, 5.0])
    umin_mod = np.array([-3.0, -3.0])
    umax_mod = np.array([3.0, 3.0])
    noise_mod = np.array([0.3, 0.3])
    x0 = np.array([0.0, 0.0, 0.0])
    x_goal = np.array([0.0, 10.0, 0.0])
    ctrl_label_1 = "Left Wheel Velocity"
    ctrl_label_2 = "Right Wheel Velocity"
    x0 = np.array([0.0, 0.0, np.pi/4])
    x_goal = np.array([10.0, 10.0, 0.0])

max_deg = 72.0 # maximum steering angle in degrees
config = MPPIConfig(
    num_samples=10000,
    horizon=40,
    dt=0.05,
    lambda_=5.0,
    
    # Cost weights
    Q=Q_mod,  # Penalize position more than heading/velocity
    Qf=Qf_mod,  # Strong terminal cost
    R=np.diag([0.1, 0.1]),  # Small control cost
    
    # Obstacle avoidance
    Q_obs=100.0,
    d_safe=env.robot_radius+0.05,  # Stay 0.05m away from obstacles
    
    # Dynamics
    dynamics_params=np.array([1.0]),  # wheelbase = 2.5m
    
    # Control limits
    u_min=umin_mod,  # max braking, max left turn
    u_max=umax_mod,     # max accel, max right turn
    
    # Sampling
    noise_sigma=noise_mod,  # acceleration noise, steering noise
)

# Create MPPI controller
mppi = MPPIBaseline(config, model.gpu, environment=env)

# ============================================================================
# Simulate
# ============================================================================

trajectory = [x0.copy()]
controls = []
num_steps = 200

x = x0.copy()

print("Running MPPI simulation...")
for step in range(num_steps):
    # Get control (returns (u_opt, is_safe))
    start = time.time()
    u, is_safe = mppi.get_control(x, x_goal)
    
    # Apply control (using simple integration - in practice, use actual dynamics)
    # Use host bicycle dynamics implementation
    x_next = model.cpu(x, u, config.dt, config.dynamics_params)
    
    x = x_next
    trajectory.append(x.copy())
    controls.append(u.copy())
    
    # Check if reached goal
    if np.linalg.norm(x[:2] - x_goal[:2]) < 0.5:
        print(f"Reached goal at step {step}")
        break
    
    if step % 20 == 0:
        end = time.time()
        print(f"Step {step}: pos=({x[0]:.2f}, {x[1]:.2f}), safe={is_safe}, time={end-start:.3f}s")

trajectory = np.array(trajectory)
controls = np.array(controls)

print(f"Simulation complete: {len(trajectory)} steps")

# ============================================================================
# Visualize
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Trajectory
ax1 = axes[0, 0]
env.visualize(ax=ax1, show_bounds=True)
ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
ax1.plot(x0[0], x0[1], 'go', markersize=12, label='Start')
ax1.plot(x_goal[0], x_goal[1], 'r*', markersize=15, label='Goal')
ax1.legend()
ax1.set_title('MPPI Trajectory with Obstacle Avoidance')
ax1.grid(True, alpha=0.3)

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
filename = f'media/mppi_result_{model_name}_{x_goal[0]}_{x_goal[1]}_{x0[2]:.2f}_static.png'
plt.savefig(filename, dpi=150)
plt.show()

print("Visualization saved to", filename)
mppi.free_gpu_buffers()