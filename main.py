from controllers.mpc_baseline import MPCBaseline, MPCConfig
from environments.staticEnv import StaticEnvironment
import numpy as np
import matplotlib.pyplot as plt

def bicycle_dynamics(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    
    px, py, theta, v = state
    a, delta = control

    L = 2.5

    px_dot = v * np.cos(theta)
    py_dot = v * np.sin(theta)
    theta_dot = v / L * np.tan(delta)
    v_dot = a

    x_next = np.array([
        px + px_dot * dt,
        py + py_dot * dt,
        theta + theta_dot * dt,
        v + v_dot * dt
    ])
    return x_next

def main():

    env = StaticEnvironment(bounds=(-2, 12, -2, 12), robot_radius=0.3)
    env.add_circle_obstacle(np.array([5.0, 5.0]), radius=1.0)
    env.add_circle_obstacle(np.array([7.0, 3.0]), radius=0.8)
    env.add_rectangle_obstacle(np.array([3.0, 8.0]), width=2.0, height=1.5)
    
    config = MPCConfig(
        horizon=20,
        dt=0.1,
        control_dim=2,
        state_dim=4,
        Q=np.eye(4),
        R=np.eye(2),
        Qf=np.eye(4) * 10,
        u_min=np.array([-1, -1]),
        u_max=np.array([1, 1])
    )

    mpc = MPCBaseline(config, bicycle_dynamics, env)
    
    # Initial state and reference trajectory
    x0 = np.array([0, 0, 0, 0])
    x_goal = np.array([10, 10, 0, 0])
    U = np.zeros((config.max_iterations, config.control_dim))
    X = np.zeros((config.max_iterations + 1, config.state_dim))
    T = np.arange(config.max_iterations) * config.dt
    X[0] = x0

    for i in range(config.max_iterations):
        # Request a safe control from the MPC (require safety)
        u, is_safe = mpc.get_control(X[i], x_goal, require_safe=True)
        
        if not is_safe:
            print(f"WARNING: Unsafe trajectory detected at iteration {i}.")
        else:
            print(f"Safe trajectory found at iteration {i}.")

        U[i] = u
        X[i+1] = bicycle_dynamics(X[i], u, config.dt)

        if np.linalg.norm(X[i+1][:2] - x_goal[:2]) < 0.5:
            print(f"Goal reached at iteration {i}.")
            break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    env.visualize(ax=ax1)
    ax1.plot(X[:i+1, 0], X[:i+1, 1], 'b.-', label='Trajectory')
    ax1.plot(x_goal[0], x_goal[1], 'ro', label='Goal')
    ax1.set_title('MPC Trajectory')
    ax1.legend()

    ax2.plot(T[:i], U[:i, 0], 'r.-', label='Acceleration')
    ax2.plot(T[:i], U[:i, 1], 'g.-', label='Steering')
    ax2.set_title('Control Inputs')
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
