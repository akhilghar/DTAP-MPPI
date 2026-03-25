from typing import Callable

from numba import cuda, jit, float32 as numba_float32
from numba.cuda.random import float32, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32
import numpy as np
import math

MAX_TERRAIN_POINTS = 100  # Must match MPPIConfig.n_terrain_points; used for shared memory allocation

# Rollout kernel for MPPI, dynamically generated based on the provided dynamics function
def make_rollout_kernel(dynamics_func, state_dim, control_dim):
    
    @cuda.jit()
    def rollout_cuda(samples, trajectories, x0, params, dt, num_samples, horizon):
        # Get the index of the current thread
        idx = cuda.grid(1)

        if idx < num_samples:
            # Initialize the trajectory with the initial state
            for i in range(state_dim):
                trajectories[idx, 0, i] = x0[i]

            for t in range(horizon):
                # Get current state and control
                x = cuda.local.array(state_dim, dtype=np.float32)
                for i in range(state_dim):
                    x[i] = trajectories[idx, t, i]

                u = cuda.local.array(control_dim, dtype=np.float32)
                for i in range(control_dim):
                    u[i] = samples[idx, t, i]

                x_next = cuda.local.array(state_dim, dtype=np.float32)

                # Compute next state using the dynamics function
                dynamics_func(x, u, dt, params, x_next)

                # Store the next state in the trajectory
                for i in range(state_dim):
                    trajectories[idx, t + 1, i] = x_next[i]

    return rollout_cuda

# Geometry helper function to compute distance from a point to a line segment (used for obstacle cost evaluation)
@cuda.jit(device=True)
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    # Compute the distance from point (px, py) to the line segment defined by (x1, y1) and (x2, y2)
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        # The segment is a point
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))  # Clamp t to the range [0, 1]

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    distance = math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
    return distance

@cuda.jit(device=True)
def is_in_polygon(px, py, poly_vertices, start, num_vertices):
    # Ray-casting algorithm to determine if point (px, py) is inside the polygon
    # defined by a contiguous block in poly_vertices starting at `start`.
    inside = False
    if num_vertices == 0:
        return False
    j = start + num_vertices - 1
    for ii in range(start, start + num_vertices):
        xi = poly_vertices[ii, 0]
        yi = poly_vertices[ii, 1]
        xj = poly_vertices[j, 0]
        yj = poly_vertices[j, 1]
        # avoid division by zero in degenerate edges
        if (yi > py) != (yj > py):
            xinters = (py - yi) * (xj - xi) / (yj - yi) + xi
            if px < xinters:
                inside = not inside
        j = ii
    return inside

@cuda.jit(device=True)
def distance_to_circle(px, py, cx, cy, radius):
    # Compute the distance from point (px, py) to the circumference of a circle with center (cx, cy) and radius
    dist_to_center = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    return max(0, dist_to_center - radius)

@cuda.jit(device=True)
def distance_to_rectangle(px, py, cx, cy, width, height):
    # Compute the distance from point (px, py) to the edges of a rectangle centered at (cx, cy) with given width and height
    left = cx - width / 2
    right = cx + width / 2
    top = cy + height / 2
    bottom = cy - height / 2

    if left <= px <= right and bottom <= py <= top:
        return 0.0  # Inside the rectangle

    dx = max(left - px, 0, px - right)
    dy = max(bottom - py, 0, py - top)
    return math.sqrt(dx * dx + dy * dy)

@cuda.jit(device=True)
def distance_to_polygon(px, py, poly_vertices, start, num_vertices):
    # Compute the distance from point (px, py) to the edges of a polygon stored in
    # poly_vertices as a contiguous block starting at `start`.
    if num_vertices == 0:
        return 1e9
    min_dist = 1e9
    for ii in range(start, start + num_vertices):
        idx1 = ii
        idx2 = start + ((ii - start + 1) % num_vertices)
        x1 = poly_vertices[idx1, 0]
        y1 = poly_vertices[idx1, 1]
        x2 = poly_vertices[idx2, 0]
        y2 = poly_vertices[idx2, 1]
        dist = point_to_segment_distance(px, py, x1, y1, x2, y2)
        if dist < min_dist:
            min_dist = dist
    return min_dist

@cuda.jit
def compute_weights_kernel(costs, weights, lambda_, num_samples):
    idx = cuda.grid(1)
    if idx < num_samples:
        min_cost = costs[0]
        for i in range(1, num_samples):
            if costs[i] < min_cost:
                min_cost = costs[i]
            
        weights[idx] = math.exp(-(costs[idx] - min_cost)/lambda_)

@cuda.jit
def normalize_weights_kernel(weights, normalized_weights, num_samples):
    idx = cuda.grid(1)
    if idx < num_samples:
        sum_weights = 0.0
        for i in range(num_samples):
            sum_weights += weights[i]
        if sum_weights > 0:
            normalized_weights[idx] = weights[idx] / sum_weights
        else:
            normalized_weights[idx] = 1.0 / num_samples  # Avoid division by zero, assign equal weights


@cuda.jit
def static_cost_min_dist_kernel(
        trajectories, samples, costs, min_dists,
        x_goal, Q_diag, R_diag, Qf_diag,
        circle_positions, circle_radii, num_circles,
        rect_positions, rect_widths, rect_heights, rect_angles, num_rects,
        poly_vertices, poly_starts, poly_lengths, num_polys,
        d_safe, Q_obs, robot_radius,
        num_samples, horizon, state_dim, control_dim, bounds):
    """
    Single-pass static obstacle kernel for the baseline controller.

    samples layout: (horizon, num_samples, control_dim)
    trajectories layout: (num_samples, horizon + 1, state_dim)
    """
    idx = cuda.grid(1)
    if idx < num_samples:
        cost = 0.0
        min_d = 1e9

        for t in range(horizon):
            state_cost = 0.0
            for i in range(state_dim):
                diff = trajectories[idx, t, i] - x_goal[i]
                state_cost += Q_diag[i] * diff * diff

            control_cost = 0.0
            for i in range(control_dim):
                u = samples[t, idx, i]
                control_cost += R_diag[i] * u * u

            cost += state_cost + control_cost

            px = trajectories[idx, t, 0]
            py = trajectories[idx, t, 1]

            for c in range(num_circles):
                obs_px = circle_positions[c, 0]
                obs_py = circle_positions[c, 1]
                dist = distance_to_circle(px, py, obs_px, obs_py, circle_radii[c])
                if dist < d_safe:
                    cost += Q_obs * (d_safe - dist + robot_radius)
                if dist < min_d:
                    min_d = dist

            for r in range(num_rects):
                dist = distance_to_rectangle(
                    px, py,
                    rect_positions[r, 0], rect_positions[r, 1],
                    rect_widths[r], rect_heights[r]
                )
                if dist < d_safe:
                    cost += Q_obs * (d_safe - dist + robot_radius)
                d_clr = dist - robot_radius
                if d_clr < min_d:
                    min_d = d_clr

            for p in range(num_polys):
                start = poly_starts[p]
                length = poly_lengths[p]
                if is_in_polygon(px, py, poly_vertices, start, length):
                    cost += Q_obs * (d_safe + robot_radius)
                    if -1.0 < min_d:
                        min_d = -1.0
                else:
                    dist = distance_to_polygon(px, py, poly_vertices, start, length)
                    if dist < d_safe:
                        cost += Q_obs * (d_safe - dist + robot_radius)
                    d_clr = dist - robot_radius
                    if d_clr < min_d:
                        min_d = d_clr

            if px < bounds[0] + robot_radius:
                cost += Q_obs ** 2 * (bounds[0] + robot_radius - px)
            if px > bounds[1] - robot_radius:
                cost += Q_obs ** 2 * (px - (bounds[1] - robot_radius))
            if py < bounds[2] + robot_radius:
                cost += Q_obs ** 2 * (bounds[2] + robot_radius - py)
            if py > bounds[3] - robot_radius:
                cost += Q_obs ** 2 * (py - (bounds[3] - robot_radius))

        terminal_cost = 0.0
        for i in range(state_dim):
            diff = trajectories[idx, horizon, i] - x_goal[i]
            terminal_cost += Qf_diag[i] * diff * diff
        cost += terminal_cost

        costs[idx] = cost
        min_dists[idx] = min_d


@cuda.jit
def expected_trajectory_kernel(trajectories, weights, out_traj, num_samples, horizon, state_dim):
    """
    Parallel reduction: one block per output element (t, s).
    Grid = (horizon+1)*state_dim blocks, Block = 256 threads.
    Each block's threads cooperatively sum over num_samples via shared memory.
    """
    output_idx = cuda.blockIdx.x
    total = (horizon + 1) * state_dim
    if output_idx >= total:
        return

    t   = output_idx // state_dim
    s   = output_idx % state_dim
    tid = cuda.threadIdx.x
    bsz = cuda.blockDim.x

    shared = cuda.shared.array(256, dtype=numba_float32)

    acc = numba_float32(0.0)
    i = tid
    while i < num_samples:
        acc += weights[i] * trajectories[i, t, s]
        i += bsz
    shared[tid] = acc
    cuda.syncthreads()

    stride = bsz >> 1
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()
        stride >>= 1

    if tid == 0:
        out_traj[t, s] = shared[0]


@cuda.jit
def expected_controls_kernel(samples, weights, out_controls, num_samples, horizon, control_dim):
    tid = cuda.grid(1)
    total = horizon * control_dim
    if tid < total:
        t = tid // control_dim
        u = tid % control_dim
        acc = 0.0
        for i in range(num_samples):
            acc += weights[i] * samples[i, t, u]
        out_controls[t, u] = acc


# ---------------------------------------------------------------------------
# Monte Carlo obstacle kernels
# ---------------------------------------------------------------------------

@cuda.jit
def obs_mc_rollout_kernel(rng_states, positions, velocities, speeds, radii,
                          xmin, xmax, ymin, ymax,
                          obs_trajs, R, N, horizon, dt, direction_change_prob):
    """
    Generate R independent stochastic rollouts of N dynamic obstacles.

    Thread layout: one thread per (rollout, obstacle) pair.
        tid = r * N + n   →   r = tid // N,  n = tid % N

    Each thread simulates obstacle n for `horizon` timesteps in rollout r,
    applying random direction changes and wall bouncing.

    Output shape: obs_trajs[r, n, k, xy]  — (R, N, horizon, 2)
    """
    tid = cuda.grid(1)
    r   = tid // N
    n   = tid  % N

    if r >= R or n >= N:
        return

    px  = positions[n, 0]
    py  = positions[n, 1]
    vx  = velocities[n, 0]
    vy  = velocities[n, 1]
    spd = speeds[n]
    rad = radii[n]

    for k in range(horizon):
        # Stochastic direction change
        if xoroshiro128p_uniform_float32(rng_states, tid) < direction_change_prob:
            angle = xoroshiro128p_uniform_float32(rng_states, tid) * 2.0 * math.pi
            vx = spd * math.cos(angle)
            vy = spd * math.sin(angle)

        px += vx * dt
        py += vy * dt

        # Wall bounce + clamp
        if px - rad < xmin or px + rad > xmax:
            vx = -vx
            if px - rad < xmin:
                px = xmin + rad
            elif px + rad > xmax:
                px = xmax - rad
        if py - rad < ymin or py + rad > ymax:
            vy = -vy
            if py - rad < ymin:
                py = ymin + rad
            elif py + rad > ymax:
                py = ymax - rad

        obs_trajs[r, n, k, 0] = px
        obs_trajs[r, n, k, 1] = py

# ---------------------------------------------------------------------------
# Coalesced-layout kernels used by MPPIDynObs
# ---------------------------------------------------------------------------

def make_rollout_kernel_coalesced(dynamics_func, state_dim, control_dim):
    """
    Like make_rollout_kernel but reads samples in (horizon, num_samples, control_dim)
    layout so that consecutive threads access consecutive memory at each timestep.
    trajectories layout unchanged: (num_samples, horizon+1, state_dim).
    """
    @cuda.jit
    def rollout_cuda_coalesced(samples, trajectories, x0, params, dt, terrain, terrain_info, num_samples, horizon):
        idx = cuda.grid(1)
        if idx < num_samples:
            for i in range(state_dim):
                trajectories[idx, 0, i] = x0[i]

            for t in range(horizon):
                x = cuda.local.array(state_dim, dtype=np.float32)
                for i in range(state_dim):
                    x[i] = trajectories[idx, t, i]

                u = cuda.local.array(control_dim, dtype=np.float32)
                for i in range(control_dim):
                    u[i] = samples[t, idx, i]      # (H, N, C) layout

                x_next = cuda.local.array(state_dim, dtype=np.float32)
                dynamics_func(x, u, dt, params, terrain, terrain_info, x_next)

                for i in range(state_dim):
                    trajectories[idx, t + 1, i] = x_next[i]

    return rollout_cuda_coalesced


@cuda.jit
def generate_samples_kernel(rng_states, samples, u_nominal, sigma, u_min, u_max,
                             num_samples, horizon, control_dim):
    """
    Generate perturbed control samples entirely on GPU.

    samples:   (horizon, num_samples, control_dim)  — coalesced layout
    u_nominal: (horizon, control_dim)
    sigma, u_min, u_max: (control_dim,)

    One thread per sample; draws horizon*control_dim normal values by
    advancing its own xoroshiro128p state repeatedly.
    """
    idx = cuda.grid(1)
    if idx < num_samples:
        for t in range(horizon):
            for d in range(control_dim):
                noise = xoroshiro128p_normal_float32(rng_states, idx)
                u = u_nominal[t, d] + sigma[d] * noise
                if u < u_min[d]:
                    u = u_min[d]
                if u > u_max[d]:
                    u = u_max[d]
                samples[t, idx, d] = u

# Terrain Awareness CUDA Kernels
@cuda.jit
def sample_terrain_kernel(rng_states, ground_truth, terrain_xy, terrain_elev,
                          px, py, theta, sensor_offset, sensor_radius, n_points, 
                          grid_origin_x, grid_origin_y, cell_size):
    idx = cuda.grid(1)
    if idx >= n_points:
        return
    
    # Compute sensed region
    cx = px + sensor_offset * math.cos(theta)
    cy = py + sensor_offset * math.sin(theta)

    # Shirley Disk Sampling
    r1 = xoroshiro128p_uniform_float32(rng_states, idx)
    r2 = xoroshiro128p_uniform_float32(rng_states, idx)
    r = math.sqrt(r1) * sensor_radius
    phi = 2.0 * math.pi * r2
    sx = cx + r * math.cos(phi)
    sy = cy + r * math.sin(phi)

    # Sample terrain elevation at (sx, sy)
    gx = (sx - grid_origin_x) / cell_size
    gy = (sy - grid_origin_y) / cell_size

    ix = int(gx)
    iy = int(gy)

    # Clamp indices to be within bounds
    nx = terrain_xy.shape[0]
    ny = terrain_xy.shape[1]
    ix = max(0, min(nx - 2, ix))
    iy = max(0, min(ny - 2, iy))

    # Fractional offsets for bilinear interpolation
    fx = gx - ix
    fy = gy - iy

    elev = (
        ground_truth[ix, iy] * (1 - fx) * (1 - fy) +
        ground_truth[ix + 1, iy] * fx * (1 - fy) +
        ground_truth[ix, iy + 1] * (1 - fx) * fy +
        ground_truth[ix + 1, iy + 1] * fx * fy
    )

    # Sensor noise model: Gaussian noise with std proportional to elevation
    depth = math.sqrt((sx - px) ** 2 + (sy - py) ** 2)
    noise_std = 0.1 * depth * depth  # 10% of depth as std
    noise = xoroshiro128p_normal_float32(rng_states, idx) * noise_std

    # Write to output arrays
    terrain_xy[idx, 0] = sx
    terrain_xy[idx, 1] = sy
    terrain_elev[idx] = elev + noise

@cuda.jit()
def sample_slope_kernel(terrain_xy, terrain_elev, n_points, cx, cy, out):
    tid = cuda.threadIdx.x
    sh_z = cuda.shared.array(MAX_TERRAIN_POINTS, dtype=numba_float32)
    sh_sx = cuda.shared.array(MAX_TERRAIN_POINTS, dtype=numba_float32)
    sh_sy = cuda.shared.array(MAX_TERRAIN_POINTS, dtype=numba_float32)
    sh_w = cuda.shared.array(MAX_TERRAIN_POINTS, dtype=numba_float32)

    if tid < n_points:
        dx = terrain_xy[tid, 0] - cx
        dy = terrain_xy[tid, 1] - cy
        d2 = dx * dx + dy * dy + 1e-12
        w = 1.0 / d2
        sh_z[tid] = terrain_elev[tid]*w
        sh_w[tid] = w
    else:
        sh_z[tid] = 0.0
        sh_w[tid] = 0.0
    cuda.syncthreads()

    # Parallel reduction to compute weighted average elevation
    stride = cuda.blockDim.x >> 1
    while stride > 0:
        if tid < stride:
            sh_z[tid] += sh_z[tid + stride]
            sh_w[tid] += sh_w[tid + stride]
        cuda.syncthreads()
        stride >>= 1

    z_est = sh_z[0] / sh_w[0] if sh_w[0] > 0 else 0.0
    W = sh_w[0]

    # Compute gradient of the weighted elevation field at (cx, cy) using finite differences
    cuda.syncthreads()
    if tid < n_points:
        dx = terrain_xy[tid, 0] - cx
        dy = terrain_xy[tid, 1] - cy
        d2 = dx * dx + dy * dy + 1e-12
        w4 = 1.0 / (d2*d2)
        dz = terrain_elev[tid] - z_est
        sh_sx[tid] = dz * w4 * dx
        sh_sy[tid] = dz * w4 * dy
    else:
        sh_sx[tid] = 0.0
        sh_sy[tid] = 0.0
    cuda.syncthreads()

    stride = cuda.blockDim.x >> 1
    while stride > 0:
        if tid < stride:
            sh_sx[tid] += sh_sx[tid + stride]
            sh_sy[tid] += sh_sy[tid + stride]
        cuda.syncthreads()
        stride >>= 1

    if tid == 0:
        out[0] = z_est
        out[1] = 2.0 * sh_sx[0] / W  # Slope in x direction
        out[2] = 2.0 * sh_sy[0] / W  # Slope in y direction

@cuda.jit
def mc_cost_and_min_dist_kernel(
        trajectories, samples, costs, min_dists,
        x_goal, Q_diag, R_diag, Qf_diag,
        circle_pred, circle_radii, num_circles, num_obs_rollouts,
        rect_positions, rect_widths, rect_heights, rect_angles, num_rects,
        poly_vertices, poly_starts, poly_lengths, num_polys,
        d_safe, Q_obs, robot_radius,
        num_samples, horizon, state_dim, control_dim, bounds,
        sensed_slope,
        sensed_cx, sensed_cy, sensed_radius,
        Q_slope, Q_elev, pz):
    """
    Single-pass kernel that computes both MPPI cost and minimum clearance.

    Replaces the mc_cost_kernel + mc_min_dist_kernel pair, halving the number
    of expensive inner-loop (R x N) passes over obstacle trajectories.

    trajectories: (num_samples, horizon+1, state_dim)
    samples:      (horizon, num_samples, control_dim)  — coalesced layout
    circle_pred:  (R, N, horizon, 2)
    """
    idx = cuda.grid(1)
    if idx < num_samples:
        cost  = 0.0
        min_d = 1e9

        for t in range(horizon):
            state_cost = 0.0
            for i in range(state_dim):
                diff = trajectories[idx, t, i] - x_goal[i]
                state_cost += Q_diag[i] * diff * diff

            control_cost = 0.0
            for i in range(control_dim):
                u = samples[t, idx, i]
                control_cost += R_diag[i] * u * u

            cost += state_cost + control_cost

            px = trajectories[idx, t, 0]
            py = trajectories[idx, t, 1]
            theta = trajectories[idx, t, 2]

            # MC circle cost (average over rollouts) + worst-case min_dist
            obs_cost = 0.0
            for r in range(num_obs_rollouts):
                for c in range(num_circles):
                    obs_px = circle_pred[r, c, t, 0]
                    obs_py = circle_pred[r, c, t, 1]
                    dist   = distance_to_circle(px, py, obs_px, obs_py, circle_radii[c])
                    if dist < d_safe:
                        obs_cost += Q_obs * (d_safe - dist + robot_radius)
                    if dist < min_d:
                        min_d = dist
            if num_obs_rollouts > 0:
                cost += obs_cost / num_obs_rollouts

            # Deterministic rectangle cost + clearance
            for r in range(num_rects):
                dist = distance_to_rectangle(px, py,
                    rect_positions[r, 0], rect_positions[r, 1],
                    rect_widths[r], rect_heights[r])
                if dist < d_safe:
                    cost += Q_obs * (d_safe - dist + robot_radius)
                d_clr = dist - robot_radius
                if d_clr < min_d:
                    min_d = d_clr

            # Deterministic polygon cost + clearance
            for p in range(num_polys):
                start  = poly_starts[p]
                length = poly_lengths[p]
                if is_in_polygon(px, py, poly_vertices, start, length):
                    cost += Q_obs * (d_safe + robot_radius)
                    if -1.0 < min_d:
                        min_d = -1.0
                else:
                    dist = distance_to_polygon(px, py, poly_vertices, start, length)
                    if dist < d_safe:
                        cost += Q_obs * (d_safe - dist + robot_radius)
                    d_clr = dist - robot_radius
                    if d_clr < min_d:
                        min_d = d_clr

            # Boundary cost
            if px < bounds[0] + robot_radius:
                cost += Q_obs * (bounds[0] + robot_radius - px)
            if px > bounds[1] - robot_radius:
                cost += Q_obs * (px - (bounds[1] - robot_radius))
            if py < bounds[2] + robot_radius:
                cost += Q_obs * (bounds[2] + robot_radius - py)
            if py > bounds[3] - robot_radius:
                cost += Q_obs * (py - (bounds[3] - robot_radius))

            # Goal reward
            if math.sqrt((px - x_goal[0]) ** 2 + (py - x_goal[1]) ** 2) < robot_radius:
                cost -= 100.0  # Reward for being within goal radius

            # Terrain cost - only add if within sensed region to avoid penalizing trajectories for unknown terrain
            d2x = px - sensed_cx
            d2y = py - sensed_cy
            if d2x * d2x + d2y * d2y < sensed_radius * sensed_radius:
                # Sample slope and elevation from sensed terrain
                z_est, slope_x, slope_y = sensed_slope
                slope_cost = Q_slope * (slope_x*slope_x + slope_y*slope_y) / horizon
                elev_cost  = Q_elev * math.fabs(z_est - pz) / horizon
                cost += slope_cost + elev_cost

                rx = -math.sin(theta)
                ry = math.cos(theta)
                roll_slope = slope_x * rx + slope_y * ry
                ROLL_SLOPE_MAX = 0.3 # Approximately sin(17 degrees), chosen based on typical robot capabilities and safety margins
                if roll_slope > ROLL_SLOPE_MAX:  # Threshold for excessive slope
                    cost += 0.9 * Q_obs * (roll_slope - ROLL_SLOPE_MAX)  # Penalize excessive slope in the direction of roll

        # Terminal cost
        terminal_cost = 0.0
        for i in range(state_dim):
            diff = trajectories[idx, horizon, i] - x_goal[i]
            terminal_cost += Qf_diag[i] * diff * diff
        cost += terminal_cost

        costs[idx]     = cost
        min_dists[idx] = min_d


@cuda.jit
def expected_controls_coalesced_kernel(samples, weights, out_controls,
                                        num_samples, horizon, control_dim):
    """
    Parallel reduction: one block per output element (t, c).
    Grid = horizon*control_dim blocks, Block = 256 threads.
    Each block's threads cooperatively sum over num_samples via shared memory.
    samples layout: (horizon, num_samples, control_dim) — coalesced stride-C access.
    """
    output_idx = cuda.blockIdx.x
    total = horizon * control_dim
    if output_idx >= total:
        return

    t   = output_idx // control_dim
    c   = output_idx % control_dim
    tid = cuda.threadIdx.x
    bsz = cuda.blockDim.x

    shared = cuda.shared.array(256, dtype=numba_float32)

    acc = numba_float32(0.0)
    i = tid
    while i < num_samples:
        acc += weights[i] * samples[t, i, c]
        i += bsz
    shared[tid] = acc
    cuda.syncthreads()

    stride = bsz >> 1
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()
        stride >>= 1

    if tid == 0:
        out_controls[t, c] = shared[0]

