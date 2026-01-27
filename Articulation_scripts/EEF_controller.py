import asyncio
import numpy as np
import omni
import carb

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction

from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)


USD_PATH = r"C:\Users\zuhai\Desktop\IRP\ur5e_robot_calibrated\ur5e_rework.usd"
ROBOT_PRIM = "/World/ur5e"

# Lula frame name for end-effector (must appear in lula_solver.get_all_frame_names())
EEF_FRAME_NAME = "wrist_3_link"

# Trajectory parameters
TRAJ_DURATION_S = 2.0     # how long to reach the target
N_WAYPOINTS = 80          # more = smoother, slower compute
MAX_QD_RAD_S = 1.2        # joint velocity clamp (rad/s)

# Debug
PRINT_EVERY_STEPS = 120
WARN_EVERY_STEPS = 60


world = None
robot = None
lula_solver = None
ik = None
physics_dt = None

# target request
requested_target_xyz = None
need_new_plan = False

# planned joint trajectory
traj_active = False
traj_times = None     # (N,)
traj_q = None         # (N, dof)
traj_qd = None        # (N, dof)
traj_t = 0.0

last_good_action = None
step_counter = 0

def to_np(x, dtype=np.float64):
    """Convert torch Tensor / list / tuple to numpy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(dtype)
    return np.asarray(x, dtype=dtype)


def smoothstep_cubic(s: float) -> float:
    """3s^2 - 2s^3 (zero vel at endpoints)"""
    return 3.0 * s * s - 2.0 * s * s * s


def clamp_qd(qd: np.ndarray, max_abs: float) -> np.ndarray:
    return np.clip(qd, -max_abs, max_abs)


def set_target_xyz(x: float, y: float, z: float):
    """
    Call this anytime to command a new Cartesian target (meters) for wrist_3_link.
    Example:
        set_target_xyz(0.45, 0.0, 0.35)
    """
    global requested_target_xyz, need_new_plan
    requested_target_xyz = np.array([x, y, z], dtype=np.float64)
    need_new_plan = True
    carb.log_info(f"[set_target_xyz] New target requested: {requested_target_xyz}")


# PLANNING: Cartesian waypoints -> IK -> joint waypoints (+ velocities)
def plan_joint_trajectory(target_xyz: np.ndarray):
    """
    Builds a joint trajectory q(t), qdot(t) by:
      1) reading current EE pose (wrist_3_link)
      2) generating smooth Cartesian waypoints
      3) IK each waypoint -> joint positions
      4) finite-difference -> joint velocities
    Returns (times, q, qd) or (None, None, None) on failure.
    """
    global robot, lula_solver, ik

    b_pos, b_quat = robot.get_world_pose()
    lula_solver.set_robot_base_pose(to_np(b_pos), to_np(b_quat))

    # Current end-effector pose (FK-like feedback)
    ee_pos, ee_quat = ik.compute_end_effector_pose(position_only=False)
    ee_pos = np.asarray(ee_pos, dtype=np.float64)

    # Current joints
    q0 = to_np(robot.get_joint_positions()).flatten()
    dof = q0.shape[0]

    # Build smooth Cartesian waypoints
    cart_wp = np.zeros((N_WAYPOINTS, 3), dtype=np.float64)
    for k in range(N_WAYPOINTS):
        s = k / (N_WAYPOINTS - 1)
        s = smoothstep_cubic(s)
        cart_wp[k] = (1 - s) * ee_pos + s * target_xyz

    # IK each waypoint -> q
    q_wp = np.zeros((N_WAYPOINTS, dof), dtype=np.float64)
    q_prev = q0.copy()
    last_success_idx = -1

    for i in range(N_WAYPOINTS):
        action, success = ik.compute_inverse_kinematics(
            target_position=cart_wp[i],
            target_orientation=None,  # position-only
            position_tolerance=None,
            orientation_tolerance=None,
        )

        if not success:
            carb.log_warn(f"[plan] IK failed at waypoint {i}/{N_WAYPOINTS-1}. Truncating trajectory.")
            break

        q = np.asarray(action.joint_positions, dtype=np.float64).flatten()

        if q.shape[0] == dof:
            q_full = q
        else:
            q_full = q_prev.copy()
            if action.joint_indices is not None:
                for j, ji in enumerate(action.joint_indices):
                    q_full[int(ji)] = q[j]
            else:
                q_full[: q.shape[0]] = q

        q_wp[i] = q_full
        q_prev = q_full
        last_success_idx = i

    if last_success_idx < 1:
        carb.log_warn("[plan] Planning failed: not enough valid IK points.")
        return None, None, None

    q_wp = q_wp[: last_success_idx + 1]
    n = q_wp.shape[0]

    times = np.linspace(0.0, TRAJ_DURATION_S, n, dtype=np.float64)

    # Velocities 
    qd_wp = np.zeros_like(q_wp)
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        if dt < 1e-9:
            qd_wp[i] = 0.0
        else:
            qd_wp[i] = (q_wp[i] - q_wp[i - 1]) / dt
            qd_wp[i] = clamp_qd(qd_wp[i], MAX_QD_RAD_S)
    qd_wp[0] = qd_wp[1].copy()

    carb.log_info(f"[plan] Planned {n} joint waypoints over {TRAJ_DURATION_S:.2f}s.")
    return times, q_wp, qd_wp


# 
# EXECUTION
def sample_piecewise_linear(times: np.ndarray, q_wp: np.ndarray, qd_wp: np.ndarray, t: float):
    """
    Piecewise-linear interpolation for positions.
    Velocities are taken from segment qd (or interpolated lightly).
    Returns (q, qd).
    """
    if t <= times[0]:
        return q_wp[0].copy(), qd_wp[0].copy()
    if t >= times[-1]:
        return q_wp[-1].copy(), qd_wp[-1].copy()

    i = int(np.searchsorted(times, t) - 1)
    t0, t1 = times[i], times[i + 1]
    q0, q1 = q_wp[i], q_wp[i + 1]
    qd0 = qd_wp[i]

    dt = t1 - t0
    if dt < 1e-9:
        return q0.copy(), np.zeros_like(q0)

    a = (t - t0) / dt
    q = (1 - a) * q0 + a * q1
    #  constant segment velocity (stable)
    qd = qd0.copy()
    return q, qd


def on_physics_step(dt: float):
    global need_new_plan, requested_target_xyz
    global traj_active, traj_times, traj_q, traj_qd, traj_t
    global last_good_action, step_counter

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        return

    step_counter += 1

    # Plan new trajectory 
    if need_new_plan and requested_target_xyz is not None:
        need_new_plan = False

        times, q_wp, qd_wp = plan_joint_trajectory(requested_target_xyz)
        if times is None:
            traj_active = False
            return

        traj_times, traj_q, traj_qd = times, q_wp, qd_wp
        traj_t = 0.0
        traj_active = True

    # Execute current trajectory
    if traj_active and traj_times is not None:
        q, qd = sample_piecewise_linear(traj_times, traj_q, traj_qd, traj_t)

        action = ArticulationAction(
            joint_positions=q,
            joint_velocities=qd,
            joint_indices=list(range(len(q))),
        )

        robot.apply_action(action)
        last_good_action = action

        traj_t += dt
        if traj_t >= float(traj_times[-1]):
            traj_active = False
            carb.log_info("[exec] Trajectory finished.")

    # Debug 
    if step_counter % PRINT_EVERY_STEPS == 0:
        msg = f"[step {step_counter}] active={traj_active}"
        if requested_target_xyz is not None:
            msg += f" target={requested_target_xyz}"
        carb.log_info(msg)


# 
# MAIN
async def main():
    global world, robot, lula_solver, ik, physics_dt

    if World.instance():
        World.instance().clear_instance()

    world = World(stage_units_in_meters=1.0, backend="torch")
    await world.initialize_simulation_context_async()
    await omni.kit.app.get_app().next_update_async()
    world.scene.add_default_ground_plane()

    # load robot
    add_reference_to_stage(USD_PATH, ROBOT_PRIM)
    await omni.kit.app.get_app().next_update_async()

    robot = SingleArticulation(prim_path=ROBOT_PRIM, name="ur5e")
    world.scene.add(robot)

    await world.reset_async(soft=False)

    physics_dt = float(world.get_physics_dt())
    carb.log_info(f"[init] Physics dt = {physics_dt}")

    # kinematics config (supported robot)
    kcfg = interface_config_loader.load_supported_lula_kinematics_solver_config("UR5e")
    lula_solver = LulaKinematicsSolver(**kcfg)

    frames = lula_solver.get_all_frame_names()
    carb.log_info(f"[init] EEF '{EEF_FRAME_NAME}' in frames? {EEF_FRAME_NAME in frames}")

    ik = ArticulationKinematicsSolver(robot, lula_solver, EEF_FRAME_NAME)

    world.add_physics_callback("coord_target_ik_traj", on_physics_step)

    await world.play_async()

    set_target_xyz(0.45, 0.0, 0.35)


# Run
asyncio.ensure_future(main())