"""
PyBulletEnv
===========

A higher-fidelity, multiphysics drop-in replacement for `TerraneousEnv`.

Design goal: preserve the EXACT interface the rest of DTAP-MPPI already consumes,
so the MPPI controller, DEM builder, traversability classifier and waypoint
selector work unchanged. Only the *plant* (true robot dynamics) and the *sensor*
(depth camera -> point cloud) are upgraded from analytic stand-ins to PyBullet.

What stays the same (inherited from TerraneousEnv):
    - `bounds`, `dx`, `dy`, `robot_radius`
    - `terrain`               : ground-truth heightmap (numpy), also read by MPPI
                                for its rollout slope sampling and by the classifier
    - `obstacles`             : the Python `Obstacle` objects (social-force motion
                                stays the authority; PyBullet just mirrors them so the
                                depth camera sees them and the robot can collide)
    - `add_obstacle`, `get_obstacle_data`, `predict_obstacle_trajectories`,
      `get_nearest_obstacle_distance`, `get_static_obstacle_costmap`, ...

What is NEW / upgraded here:
    - `generate_terrain(flat)`      : builds a finer smooth heightmap AND a matching
                                      PyBullet GEOM_HEIGHTFIELD collision body
    - `reset(x0)`                   : spawns the robot + obstacle bodies, settles them
    - `apply_robot_control(u, dt)`  : replaces `model.cpu(...)`. Advances real physics
                                      and returns the TRUE state [x, y, theta, pitch, roll]
    - `get_pointcloud(state, ...)`  : renders a depth image from the robot camera and
                                      unprojects it to a noisy world point cloud in the
                                      same dict schema as `Camera.get_point_cloud`
    - `step(dt, robot_pos)`         : moves obstacles (inherited) and syncs their poses
                                      into PyBullet
    - `render_rgb(state)`           : returns an RGB POV frame (for the POV GIF)

Two control modes (constructor `control_mode`):
    "dynamic"   -> full wheeled rigid-body physics (contact, slip, tipping). Highest
                   fidelity; the controller's ideal diff-drive model is only an
                   approximation of the plant.
    "kinematic" -> integrate the diff-drive kinematics for (x, y, yaw), then project
                   the body onto the real terrain to read z/pitch/roll and check real
                   collisions. Rock-solid stable; use it as a fast, always-drivable
                   baseline or when tuning the dynamic model.

Coordinate convention (kept consistent with the native env / MPPI):
    terrain[i, j] is the ground height at world (x, y) = (xmin + (i+0.5)*dx,
    ymin + (j+0.5)*dy). The PyBullet heightfield is registered to match, and a
    one-time ray-cast calibration removes any absolute-z offset so sensed elevation
    is directly comparable to `terrain`.
"""

import os
import numpy as np
from typing import Optional, Tuple

import pybullet as p
import pybullet_data

from environments.terraneousEnv import TerraneousEnv, ObstacleMode

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
_ROBOT_URDF = os.path.join(_ASSET_DIR, "diffdrive.urdf")

_WHEEL_RADIUS = 0.09  # must match diffdrive.urdf
_WHEEL_JOINT_NAMES = {
    "joint_fl": "L", "joint_rl": "L",   # left pair tracks v_left
    "joint_fr": "R", "joint_rr": "R",   # right pair tracks v_right
}


class PyBulletEnv(TerraneousEnv):
    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        robot_radius: float = 0.3,
        cell_size: float = 1.0,
        control_mode: str = "dynamic",
        terrain_amplitude: float = 0.35,
        terrain_smooth: float = 1.6,
        gui: bool = False,
        sim_substeps: int = 12,
        seed: Optional[int] = None,
        # camera intrinsics (mirror terrain_estimators/camera.py defaults)
        cam_focal_length: float = 0.02,
        cam_sensor_size: Tuple[float, float] = (0.04, 0.03),
        cam_image_size: Tuple[int, int] = (320, 240),
        cam_mounting_height: float = 0.3,
        cam_mounting_angle: float = 5.0,
        cam_baseline: float = 0.1,
        cam_max_range: float = 11.0,
        cam_pixel_step: int = 3,
    ):
        super().__init__(bounds=bounds, robot_radius=robot_radius)

        self.dx = float(cell_size)
        self.dy = float(cell_size)
        self.control_mode = control_mode
        self.terrain_amplitude = float(terrain_amplitude)
        self.terrain_smooth = float(terrain_smooth)
        self.sim_substeps = int(sim_substeps)
        self._rng = np.random.default_rng(seed)

        # --- camera model ---
        self.cam_focal_length = cam_focal_length
        self.cam_sensor_size = cam_sensor_size
        self.cam_image_size = cam_image_size
        self.cam_mounting_height = cam_mounting_height
        self.cam_mounting_angle = cam_mounting_angle
        self.cam_baseline = cam_baseline
        self.cam_max_range = cam_max_range
        self.cam_pixel_step = int(cam_pixel_step)
        self.cam_pixel_size = (cam_sensor_size[0] / cam_image_size[0],
                               cam_sensor_size[1] / cam_image_size[1])
        self.cam_hfov = 2.0 * np.arctan((cam_sensor_size[0] / 2.0) / cam_focal_length)
        self.cam_vfov = 2.0 * np.arctan((cam_sensor_size[1] / 2.0) / cam_focal_length)

        # --- PyBullet connection ---
        self._client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        # dt is set per control step in apply_robot_control via setTimeStep

        self.terrain_body = None
        self.robot_id = None
        self._wheel_joints = {"L": [], "R": []}
        self._obstacle_bodies = []
        self._terrain_z_offset = 0.0   # world_z = heightfield_z + offset  (calibrated)

        # kinematic-mode state (only used when control_mode == "kinematic")
        self._kin_state = None

    # ==================================================================
    # Terrain
    # ==================================================================
    def generate_terrain(self, flat: bool = False) -> None:
        """Override: build a finer smooth heightmap and a matching PyBullet heightfield."""
        from scipy.ndimage import gaussian_filter

        xmin, xmax, ymin, ymax = self.bounds
        nx = int(round((xmax - xmin) / self.dx))
        ny = int(round((ymax - ymin) / self.dy))

        if flat:
            self.terrain = np.zeros((nx, ny), dtype=np.float32)
        else:
            raw = self._rng.standard_normal((nx, ny)).astype(np.float32)
            smooth = gaussian_filter(raw, sigma=self.terrain_smooth)
            # normalize to +/- terrain_amplitude so slopes stay drivable
            smooth -= smooth.mean()
            peak = float(np.max(np.abs(smooth))) + 1e-6
            self.terrain = (smooth / peak * self.terrain_amplitude).astype(np.float32)

        self._build_heightfield()

    def set_terrain(self, heightmap: np.ndarray, upsample: int = 1) -> None:
        """Inject a precomputed heightmap (e.g. one recorded from a prior analytic run)
        and build the matching PyBullet heightfield. Used by the POV replay tool so the
        visualized terrain is exactly the one the simulation actually navigated.

        `upsample` bilinearly refines the surface for a smoother-looking render without
        changing its geometry: world extent and origin are preserved (dx shrinks by the
        same factor), so coordinate registration is unchanged.
        """
        heightmap = np.asarray(heightmap, dtype=np.float32)
        if upsample > 1:
            from scipy.ndimage import zoom
            heightmap = zoom(heightmap, upsample, order=1).astype(np.float32)
            self.dx = self.dx / upsample
            self.dy = self.dy / upsample
        self.terrain = heightmap
        self._build_heightfield()

    def _build_heightfield(self) -> None:
        nx, ny = self.terrain.shape
        xmin, xmax, ymin, ymax = self.bounds

        # PyBullet indexes heightfieldData[i + j*numRows]; rows->x, cols->y.
        height_data = self.terrain.astype(np.float64).flatten(order="F").tolist()

        col_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[self.dx, self.dy, 1.0],
            heightfieldData=height_data,
            numHeightfieldRows=nx,
            numHeightfieldColumns=ny,
            physicsClientId=self._client,
        )
        center = [(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, 0.0]
        self.terrain_body = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_shape,
            basePosition=center,
            physicsClientId=self._client,
        )
        p.changeDynamics(self.terrain_body, -1, lateralFriction=0.9,
                         restitution=0.0, physicsClientId=self._client)
        p.changeVisualShape(self.terrain_body, -1, rgbaColor=[0.55, 0.5, 0.4, 1.0],
                            physicsClientId=self._client)

        # --- Calibrate absolute z: make world height == terrain[i,j] ---
        # Ray-cast down at a few interior cell centers, compare hit z to terrain value.
        offsets = []
        for (fi, fj) in [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3), (0.5, 0.25), (0.25, 0.5)]:
            i = int(fi * nx); j = int(fj * ny)
            i = min(max(i, 0), nx - 1); j = min(max(j, 0), ny - 1)
            wx = xmin + (i + 0.5) * self.dx
            wy = ymin + (j + 0.5) * self.dy
            hit = p.rayTest([wx, wy, 100.0], [wx, wy, -100.0],
                            physicsClientId=self._client)[0]
            if hit[0] == self.terrain_body:
                offsets.append(float(self.terrain[i, j]) - hit[3][2])
        self._terrain_z_offset = float(np.mean(offsets)) if offsets else 0.0
        if abs(self._terrain_z_offset) > 1e-6:
            p.resetBasePositionAndOrientation(
                self.terrain_body,
                [center[0], center[1], center[2] + self._terrain_z_offset],
                [0, 0, 0, 1], physicsClientId=self._client)
            self._terrain_z_offset = 0.0

    def _terrain_height_at(self, x: float, y: float) -> float:
        """Bilinear ground-truth elevation from the numpy heightmap (world frame)."""
        xmin, _, ymin, _ = self.bounds
        nx, ny = self.terrain.shape
        gx = (x - xmin) / self.dx - 0.5
        gy = (y - ymin) / self.dy - 0.5
        i = int(np.floor(gx)); j = int(np.floor(gy))
        i = min(max(i, 0), nx - 2); j = min(max(j, 0), ny - 2)
        fx = np.clip(gx - i, 0.0, 1.0); fy = np.clip(gy - j, 0.0, 1.0)
        return float(
            self.terrain[i, j] * (1 - fx) * (1 - fy) +
            self.terrain[i + 1, j] * fx * (1 - fy) +
            self.terrain[i, j + 1] * (1 - fx) * fy +
            self.terrain[i + 1, j + 1] * fx * fy
        )

    # ==================================================================
    # Robot + obstacle bodies
    # ==================================================================
    def reset(self, x0: np.ndarray) -> np.ndarray:
        """Spawn the robot and obstacle bodies, settle, and return the true state."""
        self._spawn_robot(x0)
        self._spawn_obstacle_bodies()
        if self.control_mode == "dynamic":
            # let the robot settle onto the terrain
            p.setTimeStep(1.0 / 240.0, physicsClientId=self._client)
            for _ in range(120):
                self._apply_wheel_velocities(0.0, 0.0)
                p.stepSimulation(physicsClientId=self._client)
        else:
            self._kin_state = np.array([x0[0], x0[1], x0[2]], dtype=np.float64)
        return self.get_state()

    def _spawn_robot(self, x0: np.ndarray) -> None:
        px, py, theta = float(x0[0]), float(x0[1]), float(x0[2])
        gz = self._terrain_height_at(px, py)
        start_z = gz + _WHEEL_RADIUS + 0.08
        quat = p.getQuaternionFromEuler([0, 0, theta])
        self.robot_id = p.loadURDF(_ROBOT_URDF, [px, py, start_z], quat,
                                   physicsClientId=self._client)

        # map wheel joints
        self._wheel_joints = {"L": [], "R": []}
        for jidx in range(p.getNumJoints(self.robot_id, physicsClientId=self._client)):
            info = p.getJointInfo(self.robot_id, jidx, physicsClientId=self._client)
            jname = info[1].decode("utf-8")
            if jname in _WHEEL_JOINT_NAMES:
                self._wheel_joints[_WHEEL_JOINT_NAMES[jname]].append(jidx)
                p.changeDynamics(self.robot_id, jidx, lateralFriction=1.1,
                                 spinningFriction=0.006, rollingFriction=0.0,
                                 restitution=0.0, physicsClientId=self._client)
                # release default motor so our velocity controller has authority
                p.setJointMotorControl2(self.robot_id, jidx, p.VELOCITY_CONTROL,
                                        force=0.0, physicsClientId=self._client)
        p.changeDynamics(self.robot_id, -1, lateralFriction=0.3,
                         physicsClientId=self._client)

    def _spawn_obstacle_bodies(self) -> None:
        for b in self._obstacle_bodies:
            p.removeBody(b, physicsClientId=self._client)
        self._obstacle_bodies = []
        obs_height = 0.8
        for obs in self.obstacles:
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=float(obs.radius),
                                         height=obs_height, physicsClientId=self._client)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=float(obs.radius),
                                      length=obs_height,
                                      rgbaColor=[0.85, 0.15, 0.15, 1.0]
                                      if obs.mode == ObstacleMode.AVOIDANT
                                      else [0.8, 0.2, 0.8, 1.0],
                                      physicsClientId=self._client)
            gz = self._terrain_height_at(obs.position[0], obs.position[1])
            body = p.createMultiBody(
                baseMass=0.0,  # kinematic: Python social-force drives motion
                baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                basePosition=[float(obs.position[0]), float(obs.position[1]),
                              gz + obs_height / 2.0],
                physicsClientId=self._client)
            self._obstacle_bodies.append(body)

    def _sync_obstacle_bodies(self) -> None:
        for obs, body in zip(self.obstacles, self._obstacle_bodies):
            gz = self._terrain_height_at(obs.position[0], obs.position[1])
            p.resetBasePositionAndOrientation(
                body, [float(obs.position[0]), float(obs.position[1]), gz + 0.4],
                [0, 0, 0, 1], physicsClientId=self._client)

    # ==================================================================
    # State
    # ==================================================================
    def get_state(self) -> np.ndarray:
        """Return true state [px, py, theta, pitch, roll].

        pitch = tilt of the body forward axis above horizontal (uphill +),
        roll  = tilt of the body left axis above horizontal (left-up +).
        Both are read from the robot's actual orientation resting on the terrain.
        """
        if self.control_mode == "kinematic":
            px, py, theta = self._kin_state
            pitch, roll = self._terrain_pitch_roll(px, py, theta)
            return np.array([px, py, theta, pitch, roll], dtype=np.float32)

        pos, quat = p.getBasePositionAndOrientation(self.robot_id,
                                                    physicsClientId=self._client)
        rot = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float64).reshape(3, 3)
        forward = rot[:, 0]   # body +x in world
        left = rot[:, 1]      # body +y in world
        theta = float(np.arctan2(forward[1], forward[0]))
        pitch = float(np.arcsin(np.clip(forward[2], -1.0, 1.0)))
        roll = float(np.arcsin(np.clip(left[2], -1.0, 1.0)))
        return np.array([pos[0], pos[1], theta, pitch, roll], dtype=np.float32)

    def _terrain_pitch_roll(self, x: float, y: float, theta: float) -> Tuple[float, float]:
        eps = self.dx
        h_f = self._terrain_height_at(x + eps * np.cos(theta), y + eps * np.sin(theta))
        h_b = self._terrain_height_at(x - eps * np.cos(theta), y - eps * np.sin(theta))
        h_l = self._terrain_height_at(x - eps * np.sin(theta), y + eps * np.cos(theta))
        h_r = self._terrain_height_at(x + eps * np.sin(theta), y - eps * np.cos(theta))
        pitch = float(np.arctan2(h_f - h_b, 2 * eps))
        roll = float(np.arctan2(h_l - h_r, 2 * eps))
        return pitch, roll

    # ==================================================================
    # Control / stepping  (replaces model.cpu(...))
    # ==================================================================
    def _apply_wheel_velocities(self, v_right: float, v_left: float,
                                max_force: float = 25.0) -> None:
        omega_r = v_right / _WHEEL_RADIUS
        omega_l = v_left / _WHEEL_RADIUS
        for j in self._wheel_joints["R"]:
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL,
                                    targetVelocity=omega_r, force=max_force,
                                    physicsClientId=self._client)
        for j in self._wheel_joints["L"]:
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL,
                                    targetVelocity=omega_l, force=max_force,
                                    physicsClientId=self._client)

    def apply_robot_control(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Advance the plant by dt under control u = [v_right, v_left]; return true state."""
        v_right, v_left = float(u[0]), float(u[1])

        if self.control_mode == "kinematic":
            L = 2.0 * self.robot_radius
            v = 0.5 * (v_right + v_left)
            omega = (v_right - v_left) / L
            px, py, theta = self._kin_state
            self._kin_state = np.array([
                px + v * np.cos(theta) * dt,
                py + v * np.sin(theta) * dt,
                theta + omega * dt,
            ], dtype=np.float64)
            return self.get_state()

        # dynamic: substep the physics so a large control dt stays stable
        sub_dt = dt / self.sim_substeps
        p.setTimeStep(sub_dt, physicsClientId=self._client)
        self._apply_wheel_velocities(v_right, v_left)
        for _ in range(self.sim_substeps):
            p.stepSimulation(physicsClientId=self._client)
        return self.get_state()

    def step(self, dt: float, robot_pos: Optional[np.ndarray] = None) -> None:
        """Advance obstacles (inherited social-force motion) and mirror them into PyBullet."""
        super().step(dt, robot_pos=robot_pos)
        self._sync_obstacle_bodies()

    def check_for_collision(self, position: np.ndarray) -> bool:
        """Out-of-bounds or physical contact with an obstacle body."""
        if not self._in_bounds(position):
            return True
        if self.control_mode == "dynamic" and self.robot_id is not None:
            for body in self._obstacle_bodies:
                pts = p.getClosestPoints(self.robot_id, body, distance=0.0,
                                         physicsClientId=self._client)
                if pts:
                    return True
            return False
        # kinematic: analytic circle check (matches native behavior)
        return super().check_for_collision(position)

    # ==================================================================
    # Sensing: depth camera -> world point cloud
    # ==================================================================
    def _camera_pose(self, state: np.ndarray):
        px, py, theta = float(state[0]), float(state[1]), float(state[2])
        pitch = float(state[3]) if len(state) > 4 else 0.0   # >0 = nose up (uphill)
        roll = float(state[4]) if len(state) > 4 else 0.0     # >0 = left side up
        gz = self._terrain_height_at(px, py)
        eye = np.array([px, py, gz + self.cam_mounting_height], dtype=np.float64)

        # Build the camera basis from the robot's full orientation so the view
        # pitches up when climbing and BANKS with roll on side-slopes, then apply
        # the fixed mounting tilt on top.
        fwd = np.array([np.cos(theta), np.sin(theta), 0.0])
        left = np.array([-np.sin(theta), np.cos(theta), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        # pitch about the lateral axis (nose up -> look up)
        fwd, up = (fwd * np.cos(pitch) + up * np.sin(pitch),
                   -fwd * np.sin(pitch) + up * np.cos(pitch))
        # roll about the forward axis (bank the horizon)
        left, up = (left * np.cos(roll) + up * np.sin(roll),
                    -left * np.sin(roll) + up * np.cos(roll))
        # fixed camera mounting tilt (nose down by mounting_angle)
        m = np.radians(self.cam_mounting_angle)
        fwd, up = (fwd * np.cos(m) - up * np.sin(m),
                   fwd * np.sin(m) + up * np.cos(m))

        return eye, eye + fwd, up.tolist()

    def get_pointcloud(self, state: np.ndarray, noise_sigma: float = 0.1) -> dict:
        w, h = self.cam_image_size
        eye, target, up = self._camera_pose(state)
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), up,
                                   physicsClientId=self._client)
        aspect = w / h
        proj = p.computeProjectionMatrixFOV(np.degrees(self.cam_vfov), aspect,
                                            0.02, self.cam_max_range * 1.5,
                                            physicsClientId=self._client)
        img = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER,
                               flags=p.ER_NO_SEGMENTATION_MASK,
                               physicsClientId=self._client)
        depth_buffer = np.array(img[3], dtype=np.float64).reshape(h, w)

        # inverse of proj*view (both column-major length-16 -> row-major 4x4)
        V = np.array(view, dtype=np.float64).reshape(4, 4).T
        P = np.array(proj, dtype=np.float64).reshape(4, 4).T
        inv_pv = np.linalg.inv(P @ V)

        step = self.cam_pixel_step
        vs = np.arange(0, h, step)
        us = np.arange(0, w, step)
        uu, vv = np.meshgrid(us, vs)
        uu = uu.ravel(); vv = vv.ravel()
        d = depth_buffer[vv, uu]

        valid = d < 0.999  # 1.0 == background (no hit)
        uu, vv, d = uu[valid], vv[valid], d[valid]
        if uu.size == 0:
            return self._empty_pointcloud()

        ndc_x = 2.0 * (uu + 0.5) / w - 1.0
        ndc_y = 1.0 - 2.0 * (vv + 0.5) / h
        ndc_z = 2.0 * d - 1.0
        clip = np.stack([ndc_x, ndc_y, ndc_z, np.ones_like(ndc_x)], axis=0)  # (4, N)
        world_h = inv_pv @ clip                                              # (4, N)
        world = (world_h[:3] / world_h[3]).T                                 # (N, 3)

        depths = np.linalg.norm(world - eye[None, :], axis=1)
        in_range = depths <= self.cam_max_range
        world = world[in_range]; depths = depths[in_range]
        if world.shape[0] == 0:
            return self._empty_pointcloud()

        sigma, patch_size = self._point_uncertainty(depths, noise_sigma)
        world = self._add_noise(world, depths, sigma, noise_sigma)

        return {
            "points": world.astype(np.float32),
            "depths": depths.astype(np.float32),
            "sigma": sigma.astype(np.float32),
            "patch_size": patch_size.astype(np.float32),
        }

    def _point_uncertainty(self, depths, sigma):
        # mirrors terrain_estimators/camera.py::compute_point_uncertainty
        fx = self.cam_focal_length / self.cam_pixel_size[0]
        sigma_depth = sigma * (depths ** 2) / (fx * self.cam_baseline)
        incidence = np.arcsin(np.clip(self.cam_mounting_height / np.maximum(depths, 1e-3), -1, 1))
        sigma_z = sigma_depth * np.sin(incidence)
        patch_size = depths * self.cam_pixel_size[0] / self.cam_focal_length
        return sigma_z, patch_size

    def _add_noise(self, points, depths, sigma_z, sigma):
        # mirrors terrain_estimators/camera.py::add_noise
        noisy = points.copy()
        sigma_lat = sigma * depths * self.cam_pixel_size[0] / self.cam_focal_length
        noisy[:, 0] += self._rng.normal(0, np.maximum(sigma_lat, 1e-6))
        noisy[:, 1] += self._rng.normal(0, np.maximum(sigma_lat, 1e-6))
        noisy[:, 2] += self._rng.normal(0, np.maximum(sigma_z, 1e-6))
        return noisy

    @staticmethod
    def _empty_pointcloud() -> dict:
        return {
            "points": np.zeros((0, 3), dtype=np.float32),
            "depths": np.zeros(0, dtype=np.float32),
            "sigma": np.zeros(0, dtype=np.float32),
            "patch_size": np.zeros(0, dtype=np.float32),
        }

    def render_rgb(self, state: np.ndarray) -> np.ndarray:
        """Return an (H, W, 3) uint8 RGB POV frame from the robot camera."""
        w, h = self.cam_image_size
        eye, target, up = self._camera_pose(state)
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), up,
                                   physicsClientId=self._client)
        proj = p.computeProjectionMatrixFOV(np.degrees(self.cam_vfov), w / h,
                                            0.02, self.cam_max_range * 1.5,
                                            physicsClientId=self._client)
        img = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER,
                               physicsClientId=self._client)
        rgb = np.array(img[2], dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        return rgb

    def close(self) -> None:
        if self._client is not None and p.isConnected(self._client):
            p.disconnect(self._client)
            self._client = None
