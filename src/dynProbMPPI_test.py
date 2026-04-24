# test_dynamic_env.py

import numpy as np
import matplotlib.pyplot as plt
import time
from controllers.mppi_dynObs import MPPIDynObs, MPPIConfig
from controllers.waypointSelector import WaypointSelector
from dynamics.models import DYNAMICS_REGISTRY
from environments.dynamicEnv_probabilistic import ProbabilisticEnv, Obstacle, ObstacleMode
from terrain_estimators.DEM_builder import DEMBuilder
from terrain_estimators.camera import Camera
from terrain_estimators.traversability_BCM import TraversabilityClassifier, _compute_attribute_vector

# ============================================================================
# Setup Environment
# ============================================================================

# corridor_width = 4.0
env = ProbabilisticEnv(bounds=(-2, 12, -2, 12), robot_radius=0.3)
env.generate_terrain(flat=False)

# Add moving circular obstacles
rng = np.random.default_rng(seed=42)
for i in range(0,6):
    env.add_obstacle(
        Obstacle(position=[np.random.randint(2.0, 11.0), np.random.randint(2.0, 11.0)], 
                 radius=0.3+0.2*np.random.rand(),
                 velocity=[2.0*np.random.rand()-1.0, 2.0*np.random.rand()-1.0],
                 mode=ObstacleMode.AVOIDANT)
    )

# Add static circular obstacles
env.add_obstacle(
    Obstacle(position=[5.0, 5.0], 
             radius=2.0,
             velocity=[0.0, 0.0],
             mode=ObstacleMode.STATIC)
)

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
    x_goal = np.array([0.0, 20.0, np.pi/2, 0.0])

else:
    Q_mod=np.diag([1.0, 1.0, 0.5, 1.0, 2.0])
    Qf_mod=np.diag([40.0, 40.0, 0.5, 10.0, 10.0])
    R_mod = np.eye(control_dim)
    umin_mod = np.array([-3.0, -3.0])
    umax_mod = np.array([3.0, 3.0])
    noise_mod = np.array([0.65, 0.65])
    ctrl_label_1 = "Left Wheel Velocity"
    ctrl_label_2 = "Right Wheel Velocity"
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    x_goal = np.array([10.0, 10.0, 0.0, 0.0, 0.0])


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

    dynamics_params=np.array([2*env.robot_radius, 0.1]),

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
    grid_half_size=6,
    goal_weight=25.0,
    obstacle_weight=50.0,
    terrain_weight=5.0,
    heading_weight=1.0,
    d_safe=config.d_safe
)

mppi = MPPIDynObs(config, model.gpu, environment=env, dem=dem)

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
num_steps = 500
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
        scores, centers = cam.classify_point_cloud(
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
              f"online={1e3*(_t_online-_t0):.1f}ms  wp={1e3*(_t_wp-_t_online):.1f}ms  mppi={1e3*_t_mppi:.1f}ms")
        
    if step % 50 == 0 & step > 0:
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
                    color='red')
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

# ============================================================================
# Robot POV Visualization
# ============================================================================

from scipy.ndimage import zoom as _nd_zoom
from PIL import Image as _PILImage, ImageDraw as _ImageDraw


def render_robot_pov_image(robot_state, terrain, env_bounds, env_cell_size,
                            cam_mounting_height=0.3, cam_mounting_angle_deg=5.0,
                            cam_hfov_deg=90.0, cam_vfov_deg=73.74,
                            img_w=320, img_h=240, max_range=11.0, upsample=8,
                            point_radius=20*env.dx, goal_pos=None,
                            trav_overlay=None, trav_observed=None):
    # --- Upsample terrain for smoother rendering ---
    terrain_r = _nd_zoom(terrain.astype(np.float32), upsample, order=1)

    rpx, rpy = float(robot_state[0]), float(robot_state[1])
    theta = float(robot_state[2])
    pitch = float(robot_state[3]) if len(robot_state) > 4 else 0.0
    roll  = float(robot_state[4]) if len(robot_state) > 4 else 0.0

    xmin, xmax, ymin, ymax = env_bounds

    # Ground elevation under robot
    ix = int(np.clip((rpx - xmin) / env_cell_size, 0, terrain.shape[0] - 1))
    iy = int(np.clip((rpy - ymin) / env_cell_size, 0, terrain.shape[1] - 1))
    ground_z = float(terrain[ix, iy])
    cam_pos = np.array([rpx, rpy, ground_z + cam_mounting_height], dtype=np.float64)

    # --- Camera orientation in world frame ---
    # total_pitch > 0  →  camera nose points down
    total_pitch = np.radians(cam_mounting_angle_deg) + pitch

    c_th, s_th = np.cos(theta), np.sin(theta)
    c_p,  s_p  = np.cos(total_pitch), np.sin(total_pitch)
    c_r,  s_r  = np.cos(roll),        np.sin(roll)

    # Forward vector (world frame): heading theta, tilted down by total_pitch
    forward = np.array([c_th * c_p, s_th * c_p, -s_p])
    # Right vector at zero roll, then rotated by roll
    right_0 = np.array([s_th, -c_th, 0.0])
    right   = c_r * right_0 + s_r * np.cross(forward, right_0)
    right  /= np.linalg.norm(right)
    # Up = right × forward  (right-hand camera frame)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    half_w = np.tan(np.radians(cam_hfov_deg) / 2.0)
    half_h = np.tan(np.radians(cam_vfov_deg) / 2.0)

    # --- Build terrain point cloud ---
    nx_r, ny_r = terrain_r.shape
    xs = np.linspace(xmin, xmax, nx_r)
    ys = np.linspace(ymin, ymax, ny_r)
    Xg, Yg = np.meshgrid(xs, ys, indexing='ij')

    # Original-grid indices for each upsampled point (for traversability lookup)
    nx_orig, ny_orig = terrain.shape
    orig_ix_all = np.clip(np.floor((Xg.ravel() - xmin) / env_cell_size).astype(int), 0, nx_orig - 1)
    orig_iy_all = np.clip(np.floor((Yg.ravel() - ymin) / env_cell_size).astype(int), 0, ny_orig - 1)

    dxv = Xg.ravel() - cam_pos[0]
    dyv = Yg.ravel() - cam_pos[1]
    dzv = terrain_r.ravel() - cam_pos[2]

    # Project onto camera axes
    depth  = dxv * forward[0] + dyv * forward[1] + dzv * forward[2]
    r_comp = dxv * right[0]   + dyv * right[1]   + dzv * right[2]
    u_comp = dxv * up[0]      + dyv * up[1]       + dzv * up[2]

    # Keep only points in front of camera and within sensor range
    valid = (depth > 0.05) & (depth <= max_range)
    depth_v = depth[valid];  r_v = r_comp[valid]
    u_v     = u_comp[valid]; elev_v = terrain_r.ravel()[valid]
    orig_ix_v = orig_ix_all[valid]
    orig_iy_v = orig_iy_all[valid]

    # Normalised device coordinates: [-1, +1]
    u_ndc = r_v / (depth_v * half_w)   # -1=left,  +1=right
    v_ndc = u_v / (depth_v * half_h)   # -1=below, +1=above

    in_fov = (np.abs(u_ndc) <= 1.0) & (np.abs(v_ndc) <= 1.0)
    depth_v   = depth_v[in_fov];   u_ndc     = u_ndc[in_fov]
    v_ndc     = v_ndc[in_fov];     elev_v    = elev_v[in_fov]
    orig_ix_v = orig_ix_v[in_fov]; orig_iy_v = orig_iy_v[in_fov]

    # Raster coordinates
    col = np.clip(((u_ndc + 1.0) / 2.0 * (img_w - 1)).astype(int), 0, img_w - 1)
    row = np.clip(((1.0 - (v_ndc + 1.0) / 2.0) * (img_h - 1)).astype(int), 0, img_h - 1)

    # Far-to-near sort → near points overwrite far ones via numpy assignment
    order     = np.argsort(depth_v)[::-1]
    col       = col[order];       row       = row[order]
    depth_v   = depth_v[order];   elev_v    = elev_v[order]
    orig_ix_v = orig_ix_v[order]; orig_iy_v = orig_iy_v[order]

    # --- Colour: traversability tri-color (green/yellow/red) + depth fog ---
    fog     = np.clip(depth_v / max_range, 0, 1)[:, np.newaxis]
    fog_col = np.array([[155, 165, 175]], dtype=np.float32)

    if trav_overlay is not None and trav_observed is not None:
        scores = trav_overlay[orig_ix_v, orig_iy_v]
        obs    = trav_observed[orig_ix_v, orig_iy_v]
        rgb    = np.zeros((len(scores), 3), dtype=np.float32)

        # Color thresholds: green below YELLOW_THR, red above RED_THR
        YELLOW_THR = 0.35
        RED_THR    = 0.65

        # Unobserved cells → gray
        rgb[~obs] = [140.0, 140.0, 140.0]

        # Green (0,200,0) → Yellow (255,220,0)  for score in [0, YELLOW_THR]
        low = obs & (scores <= YELLOW_THR)
        t_low = scores[low] / YELLOW_THR
        rgb[low, 0] = t_low * 255.0
        rgb[low, 1] = 200.0 + t_low * 20.0   # 200 → 220
        rgb[low, 2] = 0.0

        # Yellow (255,220,0) → Red (255,0,0)  for score in (YELLOW_THR, RED_THR]
        mid = obs & (scores > YELLOW_THR) & (scores <= RED_THR)
        t_mid = (scores[mid] - YELLOW_THR) / (RED_THR - YELLOW_THR)
        rgb[mid, 0] = 255.0
        rgb[mid, 1] = (1.0 - t_mid) * 220.0
        rgb[mid, 2] = 0.0

        # Fully red for score > RED_THR
        rgb[obs & (scores > RED_THR)] = [255.0, 0.0, 0.0]

        rgb = ((1.0 - fog) * rgb + fog * fog_col).astype(np.uint8)
    else:
        # Fallback: elevation colormap
        t_min, t_max = terrain.min(), terrain.max()
        norm_e = np.clip((elev_v - t_min) / (t_max - t_min + 1e-8), 0, 1)
        cmap   = plt.get_cmap('terrain')
        rgb    = (cmap(norm_e)[:, :3] * 255).astype(np.float32)
        rgb    = ((1.0 - fog) * rgb + fog * fog_col).astype(np.uint8)

    # --- Sky gradient background ---
    img     = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    sky_top = np.array([50, 100, 170], dtype=np.float32)
    sky_bot = np.array([165, 200, 235], dtype=np.float32)
    rows_idx   = np.arange(img_h, dtype=np.float32)[:, np.newaxis, np.newaxis]  # (H, 1, 1)
    t_sky      = rows_idx / max(img_h - 1, 1)
    img[:, :] = ((1.0 - t_sky) * sky_top + t_sky * sky_bot).astype(np.uint8)  # (H, 1, 3) → (H, W, 3)

    # Paint terrain points as filled squares of size (2*point_radius+1)
    for dr in range(-point_radius, point_radius + 1):
        for dc in range(-point_radius, point_radius + 1):
            rr = np.clip(row + dr, 0, img_h - 1)
            cc = np.clip(col + dc, 0, img_w - 1)
            img[rr, cc] = rgb

    # --- Goal beacon: project goal world position into screen space ---
    if goal_pos is not None:
        gx, gy = float(goal_pos[0]), float(goal_pos[1])
        gix = int(np.clip((gx - xmin) / env_cell_size, 0, terrain.shape[0] - 1))
        giy = int(np.clip((gy - ymin) / env_cell_size, 0, terrain.shape[1] - 1))
        gz  = float(terrain[gix, giy])

        gdx = gx - cam_pos[0];  gdy = gy - cam_pos[1];  gdz = gz - cam_pos[2]
        g_depth = gdx * forward[0] + gdy * forward[1] + gdz * forward[2]
        if g_depth > 0.05:
            g_r   = gdx * right[0] + gdy * right[1] + gdz * right[2]
            g_u   = gdx * up[0]    + gdy * up[1]    + gdz * up[2]
            g_u_ndc = g_r / (g_depth * half_w)
            g_v_ndc = g_u / (g_depth * half_h)
            if abs(g_u_ndc) <= 1.0 and abs(g_v_ndc) <= 1.0:
                gc = int((g_u_ndc + 1.0) / 2.0 * (img_w - 1))
                gr = int((1.0 - (g_v_ndc + 1.0) / 2.0) * (img_h - 1))
                beacon_r = max(6, int(18.0 / max(g_depth, 1.0)))
                rr_idx, cc_idx = np.ogrid[-beacon_r:beacon_r+1, -beacon_r:beacon_r+1]
                circle_mask = rr_idx**2 + cc_idx**2 <= beacon_r**2
                ring_mask   = rr_idx**2 + cc_idx**2 <= (beacon_r - 2)**2
                for dr in range(-beacon_r, beacon_r + 1):
                    for dc in range(-beacon_r, beacon_r + 1):
                        if circle_mask[dr + beacon_r, dc + beacon_r]:
                            pr = np.clip(gr + dr, 0, img_h - 1)
                            pc = np.clip(gc + dc, 0, img_w - 1)
                            if ring_mask[dr + beacon_r, dc + beacon_r]:
                                img[pr, pc] = [0, 210, 60]   # green fill
                            else:
                                img[pr, pc] = [255, 255, 255] # white ring

    return img


print("Rendering Robot POV GIF...")
pov_gif_path = f"./media/GIFs/POV/mppi_pov_{model_name}_{t_fin:.2f}_prob.gif"

cam_hfov_deg = float(np.degrees(cam.horizontal_fov))
cam_vfov_deg = float(np.degrees(cam.vertical_fov))

pov_frames = []
for i, state in enumerate(trajectory):
    frame_img = render_robot_pov_image(
        robot_state=state,
        terrain=env.terrain,
        env_bounds=env.bounds,
        env_cell_size=env_cell_size,
        cam_mounting_height=cam.mounting_height,
        cam_mounting_angle_deg=cam.mounting_angle,
        cam_hfov_deg=cam_hfov_deg,
        cam_vfov_deg=cam_vfov_deg,
        img_w=320, img_h=240,
        max_range=cam.max_range,
        upsample=8,
        point_radius=2,
        goal_pos=x_goal[:2],
        trav_overlay=dem.traversability_overlay,
        trav_observed=dem.observed,
    )

    # HUD overlay
    pil  = _PILImage.fromarray(frame_img)
    draw = _ImageDraw.Draw(pil)
    rpx, rpy    = float(state[0]), float(state[1])
    heading_deg = float(np.degrees(state[2])) % 360.0
    dist_goal   = float(np.linalg.norm(state[:2] - x_goal[:2]))
    pitch_deg   = float(np.degrees(state[3])) if len(state) > 4 else 0.0
    roll_deg    = float(np.degrees(state[4])) if len(state) > 4 else 0.0
    draw.text((6,  4), f"Step {i:03d}",                  fill=(255, 255, 200))
    draw.text((6, 18), f"Pos  ({rpx:.1f}, {rpy:.1f})",  fill=(255, 255, 200))
    draw.text((6, 32), f"Hdg  {heading_deg:.1f}\u00b0",  fill=(255, 255, 200))
    draw.text((6, 46), f"Pitch {pitch_deg:.1f}\u00b0",   fill=(255, 255, 200))
    draw.text((6, 60), f"Roll  {roll_deg:.1f}\u00b0",    fill=(255, 255, 200))
    draw.text((6, 74), f"Goal {dist_goal:.1f} m",        fill=(255, 255, 200))
    pov_frames.append(pil)

duration_ms = max(20, int(config.dt * 1000))
pov_frames[0].save(
    pov_gif_path,
    save_all=True,
    append_images=pov_frames[1:],
    duration=duration_ms,
    loop=0,
    optimize=False,
)
print(f"Saved Robot POV GIF: {pov_gif_path}")

# Free GPU buffers to ensure clean exit and avoid memory leaks
mppi.free_gpu_buffers()