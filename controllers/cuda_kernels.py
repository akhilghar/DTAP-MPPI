from numba import cuda, jit, float32 as numba_float32
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32
import numpy as np
import math

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
def static_cost_kernel(trajectories, samples, costs, x_goal, Q_diag, R_diag, Qf_diag,
                circle_positions, circle_radii, num_circles,
                rect_positions, rect_widths, rect_heights, rect_angles, num_rects,
                poly_vertices, poly_starts, poly_lengths, num_polys,
                d_safe, Q_obs, robot_radius, num_samples, horizon, state_dim, control_dim, bounds):
    
    idx = cuda.grid(1)
    
    if idx < num_samples:
        cost = 0.0
        for t in range(horizon):
            # State and control cost
            x = trajectories[idx, t]
            u = samples[idx, t]
            state_cost = 0.0
            control_cost = 0.0
            for i in range(state_dim):
                state_cost += Q_diag[i] * (x[i] - x_goal[i]) ** 2
            for i in range(control_dim):
                control_cost += R_diag[i] * u[i] ** 2
            cost += state_cost + control_cost
            
            # Obstacle cost based on position (assuming first two dimensions are position)
            px, py = x[0], x[1]
            
            # Circle obstacles
            for c in range(num_circles):
                obs_px = circle_positions[c, 0]
                obs_py = circle_positions[c, 1]
                dist = distance_to_circle(px, py, obs_px, obs_py, circle_radii[c])
                if dist < d_safe:
                    cost += Q_obs * (d_safe - dist + robot_radius)

            # Rectangle obstacles
            for r in range(num_rects):
                dist = distance_to_rectangle(px, py, rect_positions[r, 0], rect_positions[r, 1], rect_widths[r], rect_heights[r])
                if dist < d_safe:
                    cost += Q_obs * (d_safe - dist + robot_radius)

            # Polygon obstacles
            for p in range(num_polys):
                start = poly_starts[p]
                length = poly_lengths[p]
                if is_in_polygon(px, py, poly_vertices, start, length):
                    cost += Q_obs * (d_safe + robot_radius)  # Inside polygon is very bad
                else:
                    dist = distance_to_polygon(px, py, poly_vertices, start, length)
                    if dist < d_safe:
                        cost += Q_obs * (d_safe - dist + robot_radius)
            
            # Boundary cost
            if px < bounds[0] + robot_radius:
                cost += Q_obs**2 * (bounds[0] + robot_radius - px)
            if px > bounds[1] - robot_radius:
                cost += Q_obs**2 * (px - (bounds[1] - robot_radius))
            if py < bounds[2] + robot_radius:
                cost += Q_obs**2 * (bounds[2] + robot_radius - py)
            if py > bounds[3] - robot_radius:
                cost += Q_obs**2 * (py - (bounds[3] - robot_radius))

        # Terminal state cost
        x_final = trajectories[idx, horizon]
        terminal_cost = 0.0
        for i in range(state_dim):
            terminal_cost += Qf_diag[i] * (x_final[i] - x_goal[i]) ** 2
        cost += terminal_cost
        
        costs[idx] = cost

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
def static_min_distance_kernel(trajectories, min_dists,
                        circle_positions, circle_radii, num_circles,
                        rect_positions, rect_widths, rect_heights, rect_angles, num_rects,
                        poly_vertices, poly_starts, poly_lengths, num_polys,
                        robot_radius, num_samples, horizon):
    idx = cuda.grid(1)
    if idx < num_samples:
        min_d = 1e9
        # iterate over trajectory timesteps
        for t in range(horizon + 1):
            px = trajectories[idx, t, 0]
            py = trajectories[idx, t, 1]

            # circles
            for c in range(num_circles):
                obs_px = circle_positions[c, 0]
                obs_py = circle_positions[c, 1]
                dist = distance_to_circle(px, py, obs_px, obs_py, circle_radii[c])
                if dist < min_d:
                    min_d = dist

            # rectangles
            for r in range(num_rects):
                d = distance_to_rectangle(px, py, rect_positions[r, 0], rect_positions[r, 1], rect_widths[r], rect_heights[r]) - robot_radius
                if d < min_d:
                    min_d = d

            # polygons
            for p in range(num_polys):
                start = poly_starts[p]
                length = poly_lengths[p]
                if is_in_polygon(px, py, poly_vertices, start, length):
                    d = -1.0
                else:
                    d = distance_to_polygon(px, py, poly_vertices, start, length) - robot_radius
                if d < min_d:
                    min_d = d

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


@cuda.jit
def mc_cost_kernel(trajectories, samples, costs, x_goal, Q_diag, R_diag, Qf_diag,
                   circle_pred, circle_radii, num_circles, num_obs_rollouts,
                   rect_positions, rect_widths, rect_heights, rect_angles, num_rects,
                   poly_vertices, poly_starts, poly_lengths, num_polys,
                   d_safe, Q_obs, robot_radius,
                   num_samples, horizon, state_dim, control_dim, bounds):
    """
    MPPI cost kernel with Monte Carlo obstacle evaluation.

    circle_pred has shape (R, N, horizon, 2): circle_pred[r, c, t, xy]
    The circle obstacle cost is the average over all R rollouts, giving an
    unbiased estimate of E[obstacle_cost] under the stochastic obstacle model.
    Rectangles and polygons remain deterministic.
    """
    idx = cuda.grid(1)

    if idx < num_samples:
        cost = 0.0

        for t in range(horizon):
            x = trajectories[idx, t]
            u = samples[idx, t]

            state_cost   = 0.0
            control_cost = 0.0
            for i in range(state_dim):
                state_cost += Q_diag[i] * (x[i] - x_goal[i]) ** 2
            for i in range(control_dim):
                control_cost += R_diag[i] * u[i] ** 2
            cost += state_cost + control_cost

            px = x[0]
            py = x[1]

            # MC circle cost — average over R obstacle rollouts
            obs_cost = 0.0
            for r in range(num_obs_rollouts):
                for c in range(num_circles):
                    obs_px = circle_pred[r, c, t, 0]
                    obs_py = circle_pred[r, c, t, 1]
                    dist   = distance_to_circle(px, py, obs_px, obs_py, circle_radii[c])
                    if dist < d_safe:
                        obs_cost += Q_obs * (d_safe - dist + robot_radius)
            if num_obs_rollouts > 0:
                cost += obs_cost / num_obs_rollouts

            # Deterministic rectangle cost
            for r in range(num_rects):
                dist = distance_to_rectangle(px, py,
                    rect_positions[r, 0], rect_positions[r, 1],
                    rect_widths[r], rect_heights[r])
                if dist < d_safe:
                    cost += Q_obs * (d_safe - dist + robot_radius)

            # Deterministic polygon cost
            for p in range(num_polys):
                start  = poly_starts[p]
                length = poly_lengths[p]
                if is_in_polygon(px, py, poly_vertices, start, length):
                    cost += Q_obs * (d_safe + robot_radius)
                else:
                    dist = distance_to_polygon(px, py, poly_vertices, start, length)
                    if dist < d_safe:
                        cost += Q_obs * (d_safe - dist + robot_radius)

            # Boundary cost
            if px < bounds[0] + robot_radius:
                cost += Q_obs ** 2 * (bounds[0] + robot_radius - px)
            if px > bounds[1] - robot_radius:
                cost += Q_obs ** 2 * (px - (bounds[1] - robot_radius))
            if py < bounds[2] + robot_radius:
                cost += Q_obs ** 2 * (bounds[2] + robot_radius - py)
            if py > bounds[3] - robot_radius:
                cost += Q_obs ** 2 * (py - (bounds[3] - robot_radius))

        # Terminal cost
        x_final      = trajectories[idx, horizon]
        terminal_cost = 0.0
        for i in range(state_dim):
            terminal_cost += Qf_diag[i] * (x_final[i] - x_goal[i]) ** 2
        cost += terminal_cost

        costs[idx] = cost


@cuda.jit
def mc_min_dist_kernel(trajectories, min_dists,
                       circle_pred, circle_radii, num_circles, num_obs_rollouts,
                       rect_positions, rect_widths, rect_heights, rect_angles, num_rects,
                       poly_vertices, poly_starts, poly_lengths, num_polys,
                       robot_radius, num_samples, horizon):
    """
    Per-sample minimum clearance kernel with MC obstacle evaluation.

    circle_pred has shape (R, N, horizon, 2).
    Takes the worst-case (minimum) clearance over all R rollouts, which is
    the appropriate conservative bound for safety checking.
    """
    idx = cuda.grid(1)

    if idx < num_samples:
        min_d = 1e9

        for t in range(horizon):
            px = trajectories[idx, t, 0]
            py = trajectories[idx, t, 1]

            # Worst-case over all obstacle rollouts
            for r in range(num_obs_rollouts):
                for c in range(num_circles):
                    obs_px = circle_pred[r, c, t, 0]
                    obs_py = circle_pred[r, c, t, 1]
                    dist   = distance_to_circle(px, py, obs_px, obs_py, circle_radii[c])
                    if dist < min_d:
                        min_d = dist

            for r in range(num_rects):
                d = distance_to_rectangle(px, py,
                    rect_positions[r, 0], rect_positions[r, 1],
                    rect_widths[r], rect_heights[r]) - robot_radius
                if d < min_d:
                    min_d = d

            for p in range(num_polys):
                start  = poly_starts[p]
                length = poly_lengths[p]
                if is_in_polygon(px, py, poly_vertices, start, length):
                    d = -1.0
                else:
                    d = distance_to_polygon(px, py, poly_vertices, start, length) - robot_radius
                if d < min_d:
                    min_d = d

        min_dists[idx] = min_d


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
    def rollout_cuda_coalesced(samples, trajectories, x0, params, dt, num_samples, horizon):
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
                dynamics_func(x, u, dt, params, x_next)

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


@cuda.jit
def mc_cost_and_min_dist_kernel(
        trajectories, samples, costs, min_dists,
        x_goal, Q_diag, R_diag, Qf_diag,
        circle_pred, circle_radii, num_circles, num_obs_rollouts,
        rect_positions, rect_widths, rect_heights, rect_angles, num_rects,
        poly_vertices, poly_starts, poly_lengths, num_polys,
        d_safe, Q_obs, robot_radius,
        num_samples, horizon, state_dim, control_dim, bounds):
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
