"""
pybulletPOV_replay.py
=====================

Decoupled PyBullet POV visualizer for DTAP-MPPI.

This does NOT run a simulation. It replays a run that was already produced by the
fast analytic `terraneousMPPI_test.py` (which now saves a `latest_run.npz`), and
renders a robot's-eye-view GIF from a real 3D PyBullet scene: the recorded terrain
as a heightfield, the recorded obstacles as bodies, and the camera flown along the
recorded trajectory.

Because there is no physics stepping and no in-loop rendering, your simulation runs
at exactly its native speed; the (comparatively cheap) rendering happens once here.

Usage:
    # 1) run your sim normally (unchanged) -> writes media/replays/latest_run.npz
    python -m src.terraneousMPPI_test
    # 2) render the POV GIF from that run
    python -m src.pybulletPOV_replay
    python -m src.pybulletPOV_replay --input media/replays/latest_run.npz --upsample 8
"""

import os
import sys
import time
import argparse

import numpy as np
from PIL import Image, ImageDraw

from environments.PyBulletEnv import PyBulletEnv
from environments.terraneousEnv import Obstacle, ObstacleMode


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--input", default=os.path.join(here, "..", "media", "replays", "latest_run.npz"))
    ap.add_argument("--out", default=None, help="output GIF path (default: alongside input)")
    ap.add_argument("--upsample", type=int, default=6,
                    help="bilinear terrain refinement for a smoother render (geometry unchanged)")
    ap.add_argument("--pixel-step", type=int, default=1, help="1 = full-res render")
    ap.add_argument("--image-size", type=int, nargs=2, default=[320, 240])
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--gui", action="store_true", help="watch the flythrough live")
    ap.add_argument("--stride", type=int, default=1, help="render every k-th frame")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"No run file at {args.input}. Run `python -m src.terraneousMPPI_test` first.")
        return False

    data = np.load(args.input, allow_pickle=True)
    terrain = data["terrain"]
    bounds = tuple(float(b) for b in data["bounds"])
    dx = float(data["dx"])
    robot_radius = float(data["robot_radius"])
    trajectory = data["trajectory"]
    obstacle_history = data["obstacle_history"]
    obstacle_radii = data["obstacle_radii"]
    obstacle_modes = data["obstacle_modes"]
    x_goal = data["x_goal"]

    print(f"Loaded run: {len(trajectory)} states, terrain {terrain.shape}, "
          f"{len(obstacle_radii)} obstacles")

    # ---- Build the PyBullet scene from the recorded run ----
    env = PyBulletEnv(
        bounds=bounds,
        robot_radius=robot_radius,
        cell_size=dx,
        control_mode="kinematic",       # no robot body / physics needed for POV
        gui=args.gui,
        cam_image_size=tuple(args.image_size),
        cam_pixel_step=args.pixel_step,
        cam_mounting_height=float(data["cam_mounting_height"]),
        cam_mounting_angle=float(data["cam_mounting_angle"]),
        cam_max_range=float(data["cam_max_range"]),
    )
    env.set_terrain(terrain, upsample=max(1, args.upsample))

    # Recreate obstacles (positions are driven per-frame from obstacle_history).
    for r, m in zip(obstacle_radii, obstacle_modes):
        mode = ObstacleMode(str(m)) if str(m) in {e.value for e in ObstacleMode} else ObstacleMode.APATHETIC
        env.add_obstacle(Obstacle(position=[0.0, 0.0], radius=float(r), mode=mode))
    env._spawn_obstacle_bodies()

    n_obs_frames = len(obstacle_history)
    frames = []
    t0 = time.time()
    for f in range(0, len(trajectory), args.stride):
        # place obstacles at their recorded positions for this frame
        of = min(f, n_obs_frames - 1) if n_obs_frames > 0 else None
        if of is not None:
            for i, obs in enumerate(env.obstacles):
                obs.position = np.asarray(obstacle_history[of][i], dtype=np.float32)
            env._sync_obstacle_bodies()

        state = trajectory[f]
        rgb = env.render_rgb(state)

        pil = Image.fromarray(rgb)
        d = ImageDraw.Draw(pil)
        pitch = np.degrees(state[3]) if len(state) > 4 else 0.0
        roll = np.degrees(state[4]) if len(state) > 4 else 0.0
        d.text((6, 4), f"Frame {f:03d}", fill=(255, 255, 200))
        d.text((6, 18), f"Pos ({state[0]:.1f},{state[1]:.1f})", fill=(255, 255, 200))
        d.text((6, 32), f"Hdg {np.degrees(state[2]):.0f}  Pitch {pitch:.1f}  Roll {roll:.1f}",
               fill=(255, 255, 200))
        d.text((6, 46), f"Goal ({x_goal[0]:.0f},{x_goal[1]:.0f})", fill=(255, 255, 200))
        frames.append(pil)

    render_dt = time.time() - t0
    env.close()

    out = args.out or os.path.join(os.path.dirname(args.input), "pov_replay.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=max(20, int(1000 / args.fps)), loop=0)
    print(f"Rendered {len(frames)} POV frames in {render_dt:.1f}s -> {out}")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
