import asyncio
import os
import glob
import numpy as np
import omni
import carb

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage

from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    LulaCSpaceTrajectoryGenerator,
    ArticulationTrajectory,
    interface_config_loader,
)

# =========================
# CONFIG (keep short + stable)
# =========================
USD_PATH = r"C:\Users\zuhai\Desktop\IRP\ur5e_robot_calibrated\ur5e_rework.usd"
ROBOT_PRIM = "/World/ur5e"

EEF_FRAME_NAME = "wrist_2_link"     # keep as in your version
TRAJ_DURATION_S = 10.0
N_WAYPOINTS = 25
INTERPOLATION_MODE = "cubic_spline"

PRINT_EVERY_STEPS = 30              # action log cadence
PRINT_WAYPOINTS_ONCE = True         # prints cart+joint waypoint directory once
LOG_EE_TRACKING = True              # occasional EE feedback

# If you want to make IK easier, try 15 first:
# N_WAYPOINTS = 15

# GLOBAL RUNTIME STATE

world = None
robot = None
lula_solver = None
ik_solver = None
traj_gen = None

art_traj = None
traj_t = 0.0
traj_duration = 0.0
traj_active = False

need_new_plan = False
requested_target = None
step_counter = 0

# HELPERS (minimal)
def to_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def rad_to_deg(x):
    return np.degrees(x)

def smoothstep(s):
    return 3.0 * s * s - 2.0 * s * s * s

def find_motion_generation_ext_path():
    from isaacsim.core.utils.extensions import get_extension_path_from_name
    p = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
    if p is None:
        p = get_extension_path_from_name("omni.isaac.motion_generation")
    if p is None:
        raise RuntimeError("Could not locate motion_generation extension path.")
    return p

def find_ur5e_config_paths():
    ext_path = find_motion_generation_ext_path()
    yaml_candidates = glob.glob(os.path.join(ext_path, "**", "*ur5e*robot*description*.yaml"), recursive=True)
    urdf_candidates = glob.glob(os.path.join(ext_path, "**", "*ur5e*.urdf"), recursive=True)

    if not yaml_candidates:
        yaml_candidates = glob.glob(os.path.join(ext_path, "**", "*robot_description*.yaml"), recursive=True)
        yaml_candidates = [p for p in yaml_candidates if "ur5e" in p.lower()]

    if not urdf_candidates:
        urdf_candidates = glob.glob(os.path.join(ext_path, "**", "*.urdf"), recursive=True)
        urdf_candidates = [p for p in urdf_candidates if "ur5e" in p.lower()]

    if not yaml_candidates or not urdf_candidates:
        raise RuntimeError(
            "Could not find UR5e robot_description.yaml or URDF in motion_generation extension. "
            "Hardcode paths if needed."
        )

    return sorted(yaml_candidates, key=len)[0], sorted(urdf_candidates, key=len)[0]

def get_joint_names_robust(art: SingleArticulation):
    """
    Your version does not have robot.get_joint_names().
    We try multiple ways. If all fail, return None.
    """
    # Common variants
    if hasattr(art, "get_dof_names"):
        try:
            return list(art.get_dof_names())
        except Exception:
            pass

    if hasattr(art, "dof_names"):
        try:
            return list(art.dof_names)
        except Exception:
            pass

    if hasattr(art, "get_joints_state"):
        try:
            js = art.get_joints_state()
            # Sometimes dict-like
            if isinstance(js, dict):
                for k in ("names", "joint_names", "dof_names"):
                    if k in js:
                        return list(js[k])
            # Sometimes object-like
            for k in ("names", "joint_names", "dof_names"):
                if hasattr(js, k):
                    return list(getattr(js, k))
        except Exception:
            pass

    return None

def build_cart_waypoints(start_xyz, target_xyz, n):
    cart = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        s = i / (n - 1)
        s = smoothstep(s)
        cart[i] = (1.0 - s) * start_xyz + s * target_xyz
    return cart

# TARGET API
def set_target_xyz(x, y, z):
    global requested_target, need_new_plan
    requested_target = np.array([x, y, z], dtype=np.float64)
    need_new_plan = True
    carb.log_info(f"[target] new target xyz = {requested_target}")


# PLANNING (Trajectory + ArticulationTrajectory)
def plan(target_xyz, physics_dt):
    """
    Workflow:
      1) read current q + current EE xyz
      2) build Cartesian waypoint directory (N_WAYPOINTS)
      3) IK each waypoint -> qFull waypoint (rad)
      4) map qFull -> qActive in traj_gen active joint order (robust fallback)
      5) build continuous Trajectory (10s)
      6) wrap in ArticulationTrajectory
    """
    global art_traj, traj_t, traj_duration, traj_active

    # Sync base pose to Lula
    b_pos, b_quat = robot.get_world_pose()
    lula_solver.set_robot_base_pose(to_np(b_pos), to_np(b_quat))

    # Current state
    q0 = to_np(robot.get_joint_positions()).astype(np.float64).flatten()
    ee0, _ = ik_solver.compute_end_effector_pose(position_only=True)
    ee0 = np.asarray(ee0, dtype=np.float64)

    carb.log_info("========== CONFIG ==========")
    carb.log_info(f"duration={TRAJ_DURATION_S:.3f}s  dt={float(physics_dt):.6f}s  waypoints={N_WAYPOINTS}")
    carb.log_info(f"EEF frame = {EEF_FRAME_NAME}")
    carb.log_info(f"q0(deg) = {rad_to_deg(q0)}")
    carb.log_info(f"EE0 = {ee0}  TARGET = {target_xyz}")
    carb.log_info("============================")

    # Build Cartesian directory
    cart_wp = build_cart_waypoints(ee0, target_xyz, N_WAYPOINTS)

    # IK each waypoint -> qFull
    dof = q0.shape[0]
    q_wp_full = np.zeros((N_WAYPOINTS, dof), dtype=np.float64)
    ik_ok = np.zeros((N_WAYPOINTS,), dtype=bool)

    q_prev = q0.copy()
    last_ok = -1

    for i in range(N_WAYPOINTS):
        action, success = ik_solver.compute_inverse_kinematics(
            target_position=cart_wp[i],
            target_orientation=None,
            position_tolerance=None,
            orientation_tolerance=None,
        )
        ik_ok[i] = bool(success)

        if not success:
            carb.log_warn(f"[IK] failed at waypoint {i}/{N_WAYPOINTS-1}. stop.")
            break

        q = np.asarray(action.joint_positions, dtype=np.float64).flatten()

        # Expand to full DOF if needed
        if q.shape[0] == dof:
            q_full = q
        else:
            q_full = q_prev.copy()
            if action.joint_indices is not None:
                for j, ji in enumerate(action.joint_indices):
                    q_full[int(ji)] = q[j]
            else:
                q_full[: q.shape[0]] = q

        q_wp_full[i] = q_full
        q_prev = q_full
        last_ok = i

    if last_ok < 1:
        carb.log_warn("[plan] not enough IK-success waypoints. abort.")
        traj_active = False
        art_traj = None
        return False

    # Truncate to successful set
    cart_wp = cart_wp[: last_ok + 1]
    q_wp_full = q_wp_full[: last_ok + 1]
    ik_ok = ik_ok[: last_ok + 1]

    # Map full -> active in correct order
    active_names = None
    active_count = None
    if hasattr(traj_gen, "get_active_joints"):
        active_names = traj_gen.get_active_joints()
        active_count = len(active_names)
    else:
        active_count = 6  # UR5e arm

    robot_joint_names = get_joint_names_robust(robot)

    if robot_joint_names is not None and active_names is not None:
        name_to_idx = {n: i for i, n in enumerate(robot_joint_names)}
        active_idx = []
        missing = []
        for n in active_names:
            if n in name_to_idx:
                active_idx.append(name_to_idx[n])
            else:
                missing.append(n)

        if len(missing) == 0 and len(active_idx) == active_count:
            q_wp_active = q_wp_full[:, active_idx]
        else:
            # Fallback: assume first active_count joints correspond to arm
            carb.log_warn(f"[map] could not map by names. missing={missing}. fallback to first {active_count} joints.")
            q_wp_active = q_wp_full[:, :active_count]
    else:
        # No names available: fallback
        carb.log_warn(f"[map] joint names unavailable. fallback to first {active_count} joints.")
        q_wp_active = q_wp_full[:, :active_count]

    # Build continuous trajectory for exactly TRAJ_DURATION_S
    timestamps = np.linspace(0.0, TRAJ_DURATION_S, q_wp_active.shape[0], dtype=np.float64)
    trajectory = traj_gen.compute_timestamped_c_space_trajectory(
        waypoint_positions=q_wp_active,
        timestamps=timestamps,
        interpolation_mode=INTERPOLATION_MODE,
    )
    if trajectory is None:
        carb.log_warn("[plan] compute_timestamped_c_space_trajectory returned None.")
        traj_active = False
        art_traj = None
        return False

    # Wrap with ArticulationTrajectory (Isaac API)
    art_traj = ArticulationTrajectory(robot, trajectory, float(physics_dt))
    traj_duration = float(art_traj.get_trajectory_duration())
    traj_t = 0.0
    traj_active = True

    carb.log_info(f"[plan] SUCCESS. IK_ok={ik_ok.tolist()}")
    carb.log_info(f"[plan] ArticulationTrajectory duration(API) = {traj_duration:.3f}s")
    if active_names is not None:
        carb.log_info(f"[plan] Active joints (generator order) = {active_names}")

    if PRINT_WAYPOINTS_ONCE:
        carb.log_info("========== WAYPOINT DIRECTORY ==========")
        for i, xyz in enumerate(cart_wp):
            carb.log_info(f"wp[{i:02d}] xyz = {xyz}")
        carb.log_info("---- joint waypoints active (deg) ----")
        for i, qA in enumerate(q_wp_active):
            carb.log_info(f"qA[{i:02d}] = {rad_to_deg(qA)}")
        carb.log_info("=======================================")

    return True

# EXECUTION
def on_physics_step(dt):
    global need_new_plan, traj_active, traj_t, step_counter

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        return

    if need_new_plan and requested_target is not None:
        need_new_plan = False
        ok = plan(requested_target, dt)
        if not ok:
            return

    if not traj_active or art_traj is None:
        return

    if traj_t > traj_duration:
        traj_active = False
        carb.log_info("[exec] finished: t exceeded duration.")
        return

    # Get action at time t (API)
    action = art_traj.get_action_at_time(float(traj_t))

    # Print applied action in rad + deg
    if (step_counter % PRINT_EVERY_STEPS) == 0:
        q = np.asarray(action.joint_positions, dtype=np.float64).flatten()
        carb.log_info(f"[exec] t={traj_t:.3f}s q(deg)={rad_to_deg(q)}")

        if LOG_EE_TRACKING:
            ee, _ = ik_solver.compute_end_effector_pose(position_only=True)
            carb.log_info(f"[exec] EE xyz = {np.asarray(ee)}")

    robot.apply_action(action)

    traj_t += float(dt)
    step_counter += 1

# MAIN
async def main():
    global world, robot, lula_solver, ik_solver, traj_gen

    if World.instance():
        World.instance().clear_instance()

    world = World(stage_units_in_meters=1.0, backend="torch")
    await world.initialize_simulation_context_async()
    await omni.kit.app.get_app().next_update_async()
    world.scene.add_default_ground_plane()

    add_reference_to_stage(USD_PATH, ROBOT_PRIM)
    await omni.kit.app.get_app().next_update_async()

    robot = SingleArticulation(prim_path=ROBOT_PRIM, name="ur5e")
    world.scene.add(robot)

    await world.reset_async(soft=False)
    physics_dt = float(world.get_physics_dt())
    carb.log_info(f"[init] physics_dt = {physics_dt}")

    # Lula solver for IK base sync
    kcfg = interface_config_loader.load_supported_lula_kinematics_solver_config("UR5e")
    lula_solver = LulaKinematicsSolver(**kcfg)

    frames = lula_solver.get_all_frame_names()
    carb.log_info(f"[init] EEF '{EEF_FRAME_NAME}' exists? {EEF_FRAME_NAME in frames}")
    if EEF_FRAME_NAME not in frames:
        carb.log_warn("[init] EEF frame not found in Lula frames. IK likely fails.")

    ik_solver = ArticulationKinematicsSolver(robot, lula_solver, EEF_FRAME_NAME)

    # Trajectory generator configs
    robot_desc, urdf_path = find_ur5e_config_paths()
    carb.log_info(f"[init] robot_description.yaml = {robot_desc}")
    carb.log_info(f"[init] urdf = {urdf_path}")

    traj_gen = LulaCSpaceTrajectoryGenerator(
        robot_description_path=robot_desc,
        urdf_path=urdf_path,
    )

    world.add_physics_callback("traj_api_mode_b", on_physics_step)
    await world.play_async()

    # Start with one target
    set_target_xyz(0.8, 0, 0.2)


asyncio.ensure_future(main())


