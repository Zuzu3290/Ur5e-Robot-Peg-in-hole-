import os
import asyncio
import numpy as np
import pandas as pd
import omni
import carb

from isaacsim.core.utils.stage import (
    create_new_stage_async,
    get_current_stage,
    add_reference_to_stage,
)
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

from isaacsim.sensors.physics import _sensor

from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)

# =========================
# USER CONFIG
# =========================
USD_PATH = r"C:\Users\zuhai\Desktop\IRP\ur5e_robot_calibrated\ur5e_rework.usd"
ROBOT_PRIM = "/World/ur5e"

# Lula frame name for end-effector (must appear in lula_solver.get_all_frame_names())
EEF_FRAME_NAME = "wrist_3_link"

# Contact sensor prim path (stage path)
CONTACT_SENSOR_PRIM_PATH = "/World/ur5e/wrist_3_link/ur5e_peg/Contact_Sensor"

# Output CSV paths
JOINT_CSV_PATH = r"C:\Users\zuhai\Desktop\IRP\ur5e_robot_calibrated\joint_data.csv"
CONTACT_CSV_PATH = r"C:\Users\zuhai\Desktop\IRP\ur5e_robot_calibrated\contact_sensor_data.csv"

# Trajectory parameters
TRAJ_DURATION_S = 5.0     # reach target in this time
N_WAYPOINTS = 80          # more = smoother, slower planning
MAX_QD_RAD_S = 1.2        # joint velocity clamp (rad/s)

# Debug
PRINT_EVERY_STEPS = 120

# Phase sequence (WORLD coordinates in meters)
PHASE_TARGETS = [
    ("ACT_1",    (1.45,  2.15, 1.35)),
    ("INITIAL",  (1.45,  0.00, 1.40)),
    ("ACT_2",    (1.35,  2, 1.50)),
    ("ACT_3",    (1.55,  2.4, 1.30)),
    ("INITIAL",  (1.45,  0.00, 1.40)),
]

# =========================
# GLOBALS
# =========================
world = None
robot = None
lula_solver = None
ik = None
physics_dt = None

sensor_if = None

# Requested target (cartesian)
requested_target_xyz = None
need_new_plan = False

# Planned joint trajectory
traj_active = False
traj_times = None     # (N,)
traj_q = None         # (N, dof)
traj_qd = None        # (N, dof)
traj_t = 0.0

# Logging
joint_log = []
contact_sensor_log = []
current_phase = "INIT"

# Sequencing
phase_idx = 0
sequence_done = False
logs_saved = False

step_counter = 0

EEF_QUAT_WXYZ = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)

# =========================
# HELPERS
# =========================
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


def now_time():
    return omni.timeline.get_timeline_interface().get_current_time()


def log_joint_step(step, q, qd):
    # assumes UR5e 6-DOF
    q = np.asarray(q).flatten()
    qd = np.asarray(qd).flatten()
    if q.size < 6:
        return
    joint_log.append({
        "timestamp": float(now_time()),
        "phase": str(current_phase),
        "step": int(step),
        "q0": float(q[0]), "q1": float(q[1]), "q2": float(q[2]),
        "q3": float(q[3]), "q4": float(q[4]), "q5": float(q[5]),
        "qd0": float(qd[0]), "qd1": float(qd[1]), "qd2": float(qd[2]),
        "qd3": float(qd[3]), "qd4": float(qd[4]), "qd5": float(qd[5]),
    })


def log_contact_data(contact_force, step):
    contact_sensor_log.append({
        "contact_force_N": float(contact_force),
        "timestamp": float(now_time()),
        "Step_number": int(step),
        "phase": str(current_phase),
    })


def read_contact_force():
    # returns force (float) or None
    if sensor_if is None:
        return None
    reading = sensor_if.get_sensor_reading(CONTACT_SENSOR_PRIM_PATH)
    if reading.is_valid:
        return float(reading.value)
    return None


def set_target_xyz(x: float, y: float, z: float, phase: str = "MANUAL"):
    """
    Call this anytime to command a new Cartesian target (meters) for wrist_3_link.
    Example:
        set_target_xyz(1.45, 0.0, 2, phase="ACT_1")
    """
    global requested_target_xyz, need_new_plan, current_phase
    requested_target_xyz = np.array([x, y, z], dtype=np.float64)
    need_new_plan = True
    current_phase = phase
    carb.log_info(f"[set_target_xyz] phase={current_phase} target={requested_target_xyz}")


# =========================
# PLANNING (Cartesian -> IK -> joint waypoints)
# =========================
def plan_joint_trajectory(target_xyz: np.ndarray):
    """
    Builds q(t), qdot(t) by:
      1) reading current EE pose (wrist_3_link)
      2) generating smooth Cartesian waypoints
      3) IK each waypoint -> joint positions
      4) finite-difference -> joint velocities
    Returns (times, q, qd) or (None, None, None) on failure.
    """
    global robot, lula_solver, ik

    # update base pose for Lula
    b_pos, b_quat = robot.get_world_pose()
    lula_solver.set_robot_base_pose(to_np(b_pos), to_np(b_quat))

    # current end-effector pose
    ee_pos, _ee_quat = ik.compute_end_effector_pose(position_only=False)
    ee_pos = np.asarray(ee_pos, dtype=np.float64)

    # current joints
    q0 = to_np(robot.get_joint_positions()).flatten()
    dof = q0.shape[0]

    # build smooth cartesian waypoints
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
        orientation_tolerance=1,
        )
        if not success:
            carb.log_warn(f"[plan] IK failed at waypoint {i}/{N_WAYPOINTS-1}. Truncating.")
            break

        q = np.asarray(action.joint_positions, dtype=np.float64).flatten()

        # fill full dof if indices provided
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

    # velocities via finite differences + clamp
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


# =========================
# EXECUTION (piecewise linear)
# =========================
def sample_piecewise_linear(times: np.ndarray, q_wp: np.ndarray, qd_wp: np.ndarray, t: float):
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
    qd = qd0.copy()  # stable constant segment velocity
    return q, qd


def save_logs_once():
    global logs_saved
    if logs_saved:
        return
    os.makedirs(os.path.dirname(JOINT_CSV_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CONTACT_CSV_PATH), exist_ok=True)
    pd.DataFrame(joint_log).to_csv(JOINT_CSV_PATH, sep="|", index=False)
    pd.DataFrame(contact_sensor_log).to_csv(CONTACT_CSV_PATH, sep="|", index=False)
    logs_saved = True
    carb.log_info(f"[save] Joint log -> {JOINT_CSV_PATH}")
    carb.log_info(f"[save] Contact log -> {CONTACT_CSV_PATH}")


def start_next_phase_target():
    global phase_idx, sequence_done
    if phase_idx >= len(PHASE_TARGETS):
        sequence_done = True
        return

    name, (x, y, z) = PHASE_TARGETS[phase_idx]
    phase_idx += 1
    set_target_xyz(x, y, z, phase=name)


def on_physics_step(dt: float):
    global need_new_plan, requested_target_xyz
    global traj_active, traj_times, traj_q, traj_qd, traj_t
    global step_counter, sequence_done

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        return

    step_counter += 1

    # Plan new trajectory if requested
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

        # logs
        log_joint_step(step_counter, q, qd)

        # contact sensor
        f = read_contact_force()
        if f is not None and f > 0.0:
            log_contact_data(f, step_counter)
            # optional console print:
            # print(f"[{current_phase}] step={step_counter} contact={f:.4f} N")

        traj_t += dt

        # finished this phase
        if traj_t >= float(traj_times[-1]):
            traj_active = False
            carb.log_info(f"[exec] Phase '{current_phase}' finished.")
            if not sequence_done:
                start_next_phase_target()

    # When done, save and pause once
    if sequence_done and (not traj_active):
        save_logs_once()
        world.pause()

    # debug
    if step_counter % PRINT_EVERY_STEPS == 0:
        carb.log_info(f"[step {step_counter}] phase={current_phase} active={traj_active}")


# =========================
# MAIN
# =========================
async def main():
    global world, robot, lula_solver, ik, physics_dt, sensor_if
    global phase_idx, sequence_done

    await create_new_stage_async()
    _ = get_current_stage()

    # World
    if World.instance():
        World.instance().clear_instance()

    world = World(stage_units_in_meters=1.0, backend="torch")
    await world.initialize_simulation_context_async()
    await omni.kit.app.get_app().next_update_async()
    world.scene.add_default_ground_plane()

    # Load robot
    if not os.path.isfile(USD_PATH):
        carb.log_error(f"USD file not found: {USD_PATH}")
        return

    add_reference_to_stage(USD_PATH, ROBOT_PRIM)
    await omni.kit.app.get_app().next_update_async()

    robot = SingleArticulation(prim_path=ROBOT_PRIM, name="ur5e")
    world.scene.add(robot)

    await world.reset_async(soft=False)
    await omni.kit.app.get_app().next_update_async()

    physics_dt = float(world.get_physics_dt())
    carb.log_info(f"[init] Physics dt = {physics_dt}")

    # IK solver (ArticulationKinematicsSolver wrapping Lula)
    kcfg = interface_config_loader.load_supported_lula_kinematics_solver_config("UR5e")
    lula_solver = LulaKinematicsSolver(**kcfg)

    frames = lula_solver.get_all_frame_names()
    carb.log_info(f"[init] EEF '{EEF_FRAME_NAME}' in frames? {EEF_FRAME_NAME in frames}")

    ik = ArticulationKinematicsSolver(robot, lula_solver, EEF_FRAME_NAME)

    # contact sensor interface (acquire once)
    sensor_if = _sensor.acquire_contact_sensor_interface()

    # callback
    world.add_physics_callback("combined_ik_traj_logging", on_physics_step)

    await world.play_async()
    await omni.kit.app.get_app().next_update_async()

    # start sequence
    phase_idx = 0
    sequence_done = False
    start_next_phase_target()


# Run
asyncio.ensure_future(main())
carb.log_info("Combined UR5e IK trajectory + contact logging started.")

