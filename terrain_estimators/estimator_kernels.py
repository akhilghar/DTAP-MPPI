from numba import cuda, jit
import math

@cuda.jit
def ray_to_terrain_kernel(rays_dx, rays_dy, rays_dz, cam_x, cam_y, cam_z, heightmap,
                          origin_x, origin_y, cell_size, max_range, step_size, 
                          out_x, out_y, out_z, out_depth, out_valid):
    idx = cuda.grid(1)
    if idx >= rays_dx.shape[0]:
        return
    
    t = step_size
    max_steps = int(max_range / step_size)
    out_valid[idx] = False
    
    for step in range(max_steps):
        t = (step+1) * step_size

        sx = cam_x + rays_dx[idx] * t
        sy = cam_y + rays_dy[idx] * t
        sz = cam_z + rays_dz[idx] * t

        gx = (sx - origin_x) / cell_size
        gy = (sy - origin_y) / cell_size

        x0 = int(gx)
        y0 = int(gy)
        x1 = x0 + 1
        y1 = y0 + 1

        if x0 < 0 or x1 >= heightmap.shape[1] or y0 < 0 or y1 >= heightmap.shape[0]:
            continue
        
        # Bilinear interpolation
        terrain_z = (
            heightmap[y0, x0] * (x1 - gx) * (y1 - gy) +
            heightmap[y0, x1] * (gx - x0) * (y1 - gy) +
            heightmap[y1, x0] * (x1 - gx) * (gy - y0) +
            heightmap[y1, x1] * (gx - x0) * (gy - y0)
        )

        if sz <= terrain_z:
            out_x[idx] = sx
            out_y[idx] = sy
            out_z[idx] = terrain_z
            out_depth[idx] = t
            out_valid[idx] = True
            break