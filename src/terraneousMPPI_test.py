# test_dynamic_env.py

import numpy as np
import matplotlib.pyplot as plt
import time
from controllers.mppi_terraneous import MPPITerraneous, MPPIConfig
from controllers.waypointSelector import WaypointSelector
from dynamics.models import DYNAMICS_REGISTRY
from environments.terraneousEnv import TerraneousEnv, Obstacle, ObstacleMode
from terrain_estimators.DEM_builder import DEMBuilder
from terrain_estimators.camera import Camera
from terrain_estimators.traversability_BCM import TraversabilityClassifier, _compute_attribute_vector

# ============================================================================
# Setup Environment
# ============================================================================

env_scale = 1.0
playstyle = "static"  # "static" or "dynamic"
env = TerraneousEnv(bounds=(-2*env_scale, 12*env_scale, -2*env_scale, 12*env_scale), robot_radius=0.3)
env.generate_terrain(flat=False)

# Add moving circular obstacles
rng = np.random.default_rng()
if playstyle == "dynamic":
    for i in range(0,7):
        env.add_obstacle(
            Obstacle(position=[rng.uniform(2.0, 11.0*env_scale), rng.uniform(2.0, 11.0*env_scale)], 
                    radius=(0.3+0.2*rng.random())*env_scale,
                    velocity=[2.0*rng.random()-1.0, 2.0*rng.random()-1.0],
                    mode=ObstacleMode.AVOIDANT)
        )
else:
    for i in range(0,12):
        env.add_obstacle(
            Obstacle(position=[rng.uniform(2.0, 11.0*env_scale), rng.uniform(2.0, 11.0*env_scale)], 
                    radius=(0.3+0.2*rng.random())*env_scale,
                    velocity=[0.0, 0.0],
                    mode=ObstacleMode.STATIC)
        )

# Add static circular obstacles
"""env.add_obstacle(
    Obstacle(position=[5.0*env_scale, 4.0*env_scale], 
             radius=2.0*env_scale,
             velocity=[0.0, 0.0],
             mode=ObstacleMode.STATIC)
)"""

#print("Environment Obstacles: ")
#for obs in env.obstacles:
#    print(f"  Position: {obs.position}, Radius: {obs.radius}, Velocity: {obs.velocity}, Mode: {obs.mode}")

# ============================================================================
# Configure MPPI
# ============================================================================

# Define function used, reference this function exclusively
model_name = "differential_drive"  # "differential_drive", "ackermann", "bicycle"
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
    x_goal = np.array([0.0, 20.0, np.pi/2, 0.0])*env_scale

else:
    Q_mod=np.diag([10.0, 10.0, 0.75, 1.0, 2.0])
    Qf_mod=np.diag([100.0, 100.0, 1.0, 10.0, 10.0])
    R_mod = np.eye(control_dim)
    umin_mod = np.array([-3.0, -3.0])
    umax_mod = np.array([3.0, 3.0])
    noise_mod = np.array([0.8, 0.8])
    ctrl_label_1 = "Left Wheel Velocity"
    ctrl_label_2 = "Right Wheel Velocity"
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    x_goal = np.array([10.0, 10.0, 0.0, 0.0, 0.0])*env_scale


config = MPPIConfig(
    num_samples=8500,
    horizon=40,
    dt=0.05,
    lambda_=30.0, # increase temperature for smoother trajectory

    Q=Q_mod,
    Qf=Qf_mod,
    R=R_mod,

    Q_obs=50.0,
    d_safe=env.robot_radius + 0.1,

    dynamics_params=np.array([2*env.robot_radius, 0.01]),

    u_min=umin_mod,
    u_max=umax_mod,

    noise_sigma=noise_mod,
)

env_origin = (env.bounds[0], env.bounds[2])
env_cell_size = env.dx
env_grid_size = (int((env.bounds[1] - env.bounds[0]) / env_cell_size), int((env.bounds[3] - env.bounds[2]) / env_cell_size))

classifier = TraversabilityClassifier(
    n_classes=3,
    n_attributes=8,
    buffer_size=5000,
    retrain_interval=100,
    pitch_limit=20.0,
    roll_limit=20.0,
    slip_limit=0.5
)

classifier.heightmap_bootstrap(
    heightmap=env.terrain,
    cell_size=env_cell_size,
    patch_size=3,
    sample_size=3000
)

# print(config)
cam = Camera(
    focal_length=0.02,
    sensor_size=(0.04, 0.03),
    image_size=(640, 480),
    mounting_height=0.3,
    mounting_angle=5.0,
    baseline=0.1,
    max_range=11.0
)

dem = DEMBuilder(origin=env_origin, cell_size=env_cell_size, grid_size=env_grid_size)

waypoint_selector = WaypointSelector(
    grid_resolution=0.5,
    grid_half_size=5,
    goal_weight=5.0,
    obstacle_weight=5.0,
    terrain_weight=1.25,
    heading_weight=0.5,
    d_safe=config.d_safe
)

mppi = MPPITerraneous(config, model.gpu, environment=env, dem=dem)

# ============================================================================
# Simulation
# ============================================================================

trajectory = [x0.copy()]
controls = []
cov_log = []
subgoal_log = []

obstacle_history = []
rollout_snapshots = {}  # step -> (expected_traj, sample_trajs), sampled every 20 steps
terrain_snapshots = {}  # step -> (terrain_xy, terrain_elev, sensed_slope, sensed_center), sampled every step

x = x0.copy()
num_steps = 700
num_safe = 0
goal_reached = False

perception_interval = 1  # steps

print("Running Dynamic MPPI Simulation...")
sim_start_time = time.time()
for step in range(num_steps):
    # --- Step environment first (obstacles move) ---
    env.step(config.dt, robot_pos=x[:2])
    obstacle_history.append(
        np.array([obs.position.copy() for obs in env.obstacles])
    )

    _t0 = time.perf_counter()

    if step % perception_interval == 0:
        point_cloud = cam.get_point_cloud(
            robot_position=x[:2],
            robot_heading=x[2],
            d_heightmap=env.terrain,
            heightmap_origin=env_origin,
            heightmap_cell_size=env_cell_size,
            noise_sigma=0.1
        )
        _t_pcl = time.perf_counter()
        dem.fuse_point_cloud(point_cloud)
        _t_fuse = time.perf_counter()

        # classify point cloud
        classify_cell_size = 0.5
        scores, centers, labels = cam.classify_point_cloud(
            point_cloud=point_cloud,
            classifier=classifier,
            cell_size=classify_cell_size
        )
        _t_classify = time.perf_counter()

        # Add classification costs to DEM cost grid
        for i in range(len(centers)):
            r,c = dem.world_to_grid(centers[i])
            if dem.point_in_bounds(r,c):
                dem.traversability_overlay[r, c] = scores[i]  # weight for traversability cost
                dem.class_overlay[r, c] = labels[i]

        if step % 20 == 0:
            print(f"  [perception] pcl={1e3*(_t_pcl-_t0):.1f}ms  fuse={1e3*(_t_fuse-_t_pcl):.1f}ms  "
                  f"classify={1e3*(_t_classify-_t_fuse):.1f}ms ({len(centers)} cells)")

    # Online Learning for Classifier
    r,c = dem.world_to_grid(x[:2])
    patch_radius = 3
    if (r >= patch_radius and r < dem.grid_size[0] - patch_radius and
        c >= patch_radius and c < dem.grid_size[1] - patch_radius):
        # Perform online learning update here
        local_observations = dem.observed[r-patch_radius:r+patch_radius+1, c-patch_radius:c+patch_radius+1]
        if np.mean(local_observations) > 0.5:
            patch = dem.elevation[r-patch_radius:r+patch_radius+1, c-patch_radius:c+patch_radius+1]
            patch_r, patch_c = patch.shape
            points = np.zeros((patch_r*patch_c, 3), dtype=np.float32)

            idx = 0
            for i in range(patch_r):
                for j in range(patch_c):
                    points[idx] = np.array([
                        (c - patch_radius + j) * dem.cell_size + dem.origin[0],
                        (r - patch_radius + i) * dem.cell_size + dem.origin[1],
                        patch[i, j]
                    ])
                    idx += 1
            
            attr = _compute_attribute_vector(points, len(points), float(len(points)))

            v_cmd = x[3] if state_dim == 5 else 0.5*controls[-1][0] + 0.5*controls[-1][1]
            classifier.record_experience(
                attributes=attr,
                pitch=x[3] if state_dim == 5 else 0.0,
                roll=x[4] if state_dim == 5 else 0.0,
                desired_vel=v_cmd,
                actual_vel=v_cmd,
            )
    _t_online = time.perf_counter()

    # --- Waypoint Selection ---
    obs_positions = np.array([obs.position for obs in env.obstacles])
    obs_radii = np.array([obs.radius for obs in env.obstacles])

    dist_to_goal = np.linalg.norm(x[:2] - x_goal[:2])
    if dist_to_goal < waypoint_selector.grid_half_size * waypoint_selector.grid_resolution:
        subgoal = x_goal[:2]
    else:
        subgoal = waypoint_selector.plan_step(
            robot_pos=x[:2],
            robot_heading=x[2],
            goal_pos=x_goal[:2],
            obs_positions=obs_positions,
            obs_radii=obs_radii,
            terrain_cost_fn=dem.get_cost_at_points
        )
    _t_wp = time.perf_counter()

    subgoal_log.append(subgoal)
    mppi_target = x_goal.copy()
    mppi_target[:2] = subgoal

    # --- Get MPPI control ---
    start = time.time()
    x_query = x.copy()
    u, is_safe = mppi.get_control(x_query, mppi_target, require_safe=True)
    if is_safe:
        num_safe += 1

    cov = mppi.get_covariance()
    end = time.time()

    # --- Apply dynamics ---
    x = model.cpu(x, u, config.dt, config.dynamics_params, env.terrain, mppi.terrain_info)

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
    if (np.linalg.norm(x[:2] - x_goal[:2]) < env.robot_radius):
        print(f"Goal reached at step {step}!")
        goal_reached = True
        break

    if step % 20 == 0:
        rollout_snapshots[step] = mppi.get_rollout_snapshot(n=50)
        _t_mppi = end - start
        print(f"Step {step}: pos=({x[0]:.2f},{x[1]:.2f}), "
              f"Subgoal=({subgoal[0]:.2f},{subgoal[1]:.2f}), "
              f"position_error={np.linalg.norm(x[:2]-x_goal[:2]):.2f}, "
              f"safe={is_safe}, "
              f"online={1e3*(_t_online-_t0):.1f}ms mppi={1e3*_t_mppi:.1f}ms")
        
    if (step % 50 == 0) and (step > 0):
        recent_disp = np.linalg.norm(trajectory[-1][:2] - trajectory[-50][:2])
        if recent_disp < 0.2:
            mppi.reset_warm_start()

trajectory = np.array(trajectory)
controls = np.array(controls)
obstacle_history = np.array(obstacle_history)
subgoal_log = np.array(subgoal_log)
sim_end_time = time.time()

print(f"Simulation complete: {len(trajectory)} steps")
print(f"Safe Trajectory Rate: {num_safe/len(trajectory):.2f}")
print(f"Total Simulation Time: {sim_end_time - sim_start_time:.2f} seconds")
print(f"Goal Reached: {goal_reached}")
cov_log = np.array(cov_log)

# ============================================================================
# Record run for PyBullet POV replay  (decoupled visualizer — see
# src/pybulletPOV_replay.py). This only *saves* data; it does not affect the sim.
# ============================================================================
import os as _os
_replay_dir = _os.path.join(_os.path.dirname(__file__), "..", "media", "replays")
_os.makedirs(_replay_dir, exist_ok=True)
_replay_path = _os.path.join(_replay_dir, "latest_run.npz")
np.savez_compressed(
    _replay_path,
    terrain=env.terrain,
    bounds=np.array(env.bounds, dtype=np.float32),
    dx=np.float32(env.dx),
    robot_radius=np.float32(env.robot_radius),
    trajectory=trajectory,
    obstacle_history=obstacle_history,
    obstacle_radii=np.array([o.radius for o in env.obstacles], dtype=np.float32),
    obstacle_modes=np.array([o.mode.value for o in env.obstacles]),
    x_goal=x_goal.astype(np.float32),
    cam_mounting_height=np.float32(cam.mounting_height),
    cam_mounting_angle=np.float32(cam.mounting_angle),
    cam_max_range=np.float32(cam.max_range),
)
print(f"Saved run for PyBullet POV replay: {_replay_path}")

# ============================================================================
# Visualization
# ============================================================================

import matplotlib.animation as animation
from matplotlib import colors
from matplotlib.patches import Circle, Rectangle

fig, ax = plt.subplots(figsize=(9, 8))
ax.set_aspect('equal')

xmin, xmax, ymin, ymax = env.bounds
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_title('Robot Traversal + Sensed Terrain Overlay (per-frame updates)')
ax.set_xlabel('x')
ax.set_ylabel('y')

if env.terrain is not None:
    terrain_min = float(np.min(env.terrain))
    terrain_max = float(np.max(env.terrain))
    if terrain_min < 0.0 < terrain_max:
        terrain_norm = colors.TwoSlopeNorm(vmin=terrain_min, vcenter=0.0, vmax=terrain_max)
    else:
        terrain_norm = colors.Normalize(vmin=terrain_min, vmax=terrain_max)

    terrain_map = ax.imshow(
        env.terrain.T,
        extent=(xmin, xmax, ymin, ymax),
        origin='lower',
        cmap='terrain',
        norm=terrain_norm,
        alpha=0.65,
        interpolation='bilinear',
        zorder=0,
    )
    colorbar = fig.colorbar(terrain_map, ax=ax, pad=0.02, shrink=0.85)
    colorbar.set_label('Terrain elevation')
    sensed_terrain_norm = terrain_norm
else:
    sensed_terrain_norm = colors.Normalize(vmin=0.0, vmax=1.0)

terrain_scatter = ax.scatter(
    [], [],
    c=[],
    cmap='terrain',
    norm=sensed_terrain_norm,
    s=12,
    alpha=0.75,
    zorder=5,
)

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
                    color='red' if env.obstacles[i].mode == ObstacleMode.AVOIDANT else 'magenta')
    ax.add_patch(circle)
    obstacle_patches.append(circle)

# Trajectory trail
traj_line, = ax.plot([], [], 'b-', linewidth=2)

# Rollout visualization: sample trajectories + selected (weighted-mean) trajectory
N_DISPLAY_ROLLOUTS = 50
sample_rollout_lines = [
    ax.plot([], [], color='orange', alpha=0.25, linewidth=1.5, zorder=1)[0]
    for _ in range(N_DISPLAY_ROLLOUTS)
]
selected_rollout_line, = ax.plot([], [], color='lime', alpha=0.85, linewidth=1.5, zorder=2)

# Goal
ax.plot(x_goal[0], x_goal[1], 'r*', markersize=15)

# Subgoal
subgoal_marker, = ax.plot([], [], 'g*', markersize=12, zorder=4)

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

    # Update subgoal marker
    if frame < len(subgoal_log):
        subgoal = subgoal_log[frame]
        subgoal_marker.set_data([subgoal[0]], [subgoal[1]])

    return ([robot_patch, traj_line, heading_line, selected_rollout_line,
             terrain_scatter, subgoal_marker]
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
ani.save(f"./media/GIFs/BirdsEyeView/mppi_BEV_{model_name}_{t_fin:.2f}_prob.gif", writer="pillow", fps=1/config.dt)
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
ax3.set_ylabel("Orientation Angle (deg)")
if state_dim == 4:
    ax3.plot(time_vec, np.degrees(controls[:, 1]), label='Steering Angle')
    ax3.set_title("Steering Angle")
    ax3.grid(True)
else:
    ax3.plot(time_vec, np.degrees(trajectory[:-1, 2]), label='Steering Angle')
    ax3.plot(time_vec, np.degrees(trajectory[:-1, 3]), label='Pitch Angle')
    ax3.plot(time_vec, np.degrees(trajectory[:-1, 4]), label='Roll Angle')
    ax3.legend()
    ax3.set_title("Orientation Angles")
    ax3.grid(True)

plt.tight_layout()
filename = f'./media/Visualizations/Physics_Results/mppi_result_{model_name}_{t_fin:.2f}_dynProb.png'
plt.savefig(filename, dpi=150)
# plt.show()

print("Data Visualization Saved.")

# DEM Visualization
nx, ny = dem.grid_size
x_coords = np.linspace(env.bounds[0], env.bounds[1], nx)
y_coords = np.linspace(env.bounds[2], env.bounds[3], ny)
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

fig = plt.figure(figsize=(12, 7))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_title("Sensed Terrain Elevation")
ax1.plot_surface(X, Y, dem.elevation, cmap="terrain", alpha=0.75)
ax1.set_zlim(np.min(env.terrain)-1, np.max(env.terrain)+7)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_title("Ground Truth Terrain")
ax2.plot_surface(X, Y, env.terrain.T, cmap="terrain", alpha=0.75)
ax2.set_zlim(np.min(env.terrain)-1, np.max(env.terrain)+7)
plt.tight_layout()
dem_filename = f'./media/Visualizations/DEM_rendering/observed_dem_{t_fin:.2f}.png'
plt.savefig(dem_filename, dpi=150)

fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title("Traversability Overlay")
im1 = ax1.imshow(dem.traversability_overlay.T, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='Reds', alpha=0.75)
plt.colorbar(im1, label='Traversability Cost', ax=ax1)
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title("Traversability Overlay (Observed Cells Only)")
observed_overlay = np.where(dem.observed, dem.traversability_overlay, np.nan)
im2 = ax2.imshow(observed_overlay.T, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='Reds', alpha=0.75)
plt.colorbar(im2, label='Traversability Cost', ax=ax2)
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title("Sensed Terrain Confidence")
im3 = ax3.imshow(dem.confidence.T, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='plasma', alpha=0.75)
plt.colorbar(im3, label='Confidence', ax=ax3)
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title("Sensed Terrain Square Error")
im4 = ax4.imshow(np.square(dem.elevation.T-env.terrain.T), extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis', alpha=0.75)
plt.colorbar(im4, label='Error', ax=ax4)
plt.tight_layout()
overlay_filename = f'./media/Visualizations/costmaps/traversability_overlay_{t_fin:.2f}.png'
plt.savefig(overlay_filename, dpi=150)
print("DEM and Traversability Visualizations Saved.")

# Free GPU buffers to ensure clean exit and avoid memory leaks
mppi.free_gpu_buffers()