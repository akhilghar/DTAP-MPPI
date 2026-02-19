from numba import cuda, jit
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
def cost_kernel(trajectories, samples, costs, x_goal, Q_diag, R_diag, Qf_diag,
                circle_pred, circle_radii, num_circles,
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
                obs_px = circle_pred[c, t, 0]
                obs_py = circle_pred[c, t, 1]
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
def min_distance_kernel(trajectories, min_dists,
                        circle_pred, circle_radii, num_circles,
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
                obs_px = circle_pred[c, t, 0]
                obs_py = circle_pred[c, t, 1]
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
    tid = cuda.grid(1)
    total = (horizon + 1) * state_dim
    if tid < total:
        t = tid // state_dim
        s = tid % state_dim
        acc = 0.0
        for i in range(num_samples):
            acc += weights[i] * trajectories[i, t, s]
        out_traj[t, s] = acc


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
