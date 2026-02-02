import asyncio
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import omni
import carb
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver, interface_config_loader

#CONFIG
USD_PATH = r"C:\Users\zuhai\Desktop\IRP\ur5e_robot_calibrated\ur5e_rework.usd"
ROBOT_PRIM = "/World/ur5e"
EEF_FRAME_NAME = "wrist_2_link"
TRAJ_DURATION_S = 10.0   
N_WAYPOINTS = 25         
MAX_QD_RAD_S = 1.2       
PRINT_PLAN_ONCE = True
LOG_EE_TRACKING = True   
# Debug cadence
PRINT_EVERY_STEPS = 120

# Helpers
def to_np(x, dtype=np.float64):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(dtype)
    return np.asarray(x, dtype=dtype)

def smoothstep_cubic(s: float) -> float:
    # 3s^2 - 2s^3
    return 3.0 * s * s - 2.0 * s * s * s

def clamp_qd(qd: np.ndarray, max_abs: float) -> np.ndarray:
    return np.clip(qd, -max_abs, max_abs)

# STRUCTURES
class TrajectoryPlan:
    duration_s: float
    dt_physics: float
    eef_frame: str

    cart_waypoints: np.ndarray     # (N, 3)
    times: np.ndarray             # (N,)
    q_waypoints: np.ndarray        # (N, dof)
    qd_waypoints: np.ndarray       # (N, dof)
    ik_success: np.ndarray         # (N,) bool

# PLANNER
class TrajectoryPlanner:
    """
    Workflow:
      A) read current robot state + current EE pose
      B) generate Cartesian waypoints (N)
      C) IK each waypoint -> q_waypoints
      D) build time array + qdot via finite differences
    """
    def __init__(
        self,
        robot: SingleArticulation,
        lula_solver: LulaKinematicsSolver,
        ik: ArticulationKinematicsSolver,
        eef_frame: str,
        duration_s: float,
        n_waypoints: int,
        max_qd: float,
    ):
        self.robot = robot
        self.lula_solver = lula_solver
        self.ik = ik
        self.eef_frame = eef_frame
        self.duration_s = float(duration_s)
        self.n_waypoints = int(n_waypoints)
        self.max_qd = float(max_qd)

    def _sync_lula_base_pose(self):
        b_pos, b_quat = self.robot.get_world_pose()
        self.lula_solver.set_robot_base_pose(to_np(b_pos), to_np(b_quat))

    def _get_current_joint_positions(self) -> np.ndarray:
        return to_np(self.robot.get_joint_positions()).flatten()

    def _get_current_ee_position(self) -> np.ndarray:
        ee_pos, _ = self.ik.compute_end_effector_pose(position_only=True)
        return np.asarray(ee_pos, dtype=np.float64)

    def _build_cartesian_waypoints(self, start_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
        cart = np.zeros((self.n_waypoints, 3), dtype=np.float64)
        for k in range(self.n_waypoints):
            s = k / (self.n_waypoints - 1)
            s = smoothstep_cubic(s)
            cart[k] = (1 - s) * start_xyz + s * target_xyz
        return cart

    def plan(self, target_xyz: np.ndarray, dt_physics: float) -> Optional[TrajectoryPlan]:
        self._sync_lula_base_pose()

        q0 = self._get_current_joint_positions()
        dof = q0.shape[0]
        start_xyz = self._get_current_ee_position()
        cart_wp = self._build_cartesian_waypoints(start_xyz, target_xyz)
        q_wp = np.zeros((self.n_waypoints, dof), dtype=np.float64)
        ik_ok = np.zeros((self.n_waypoints,), dtype=bool)

        q_prev = q0.copy()
        for i in range(self.n_waypoints):
            action, success = self.ik.compute_inverse_kinematics(
                target_position=cart_wp[i],
                target_orientation=None,     # position-only IK
                position_tolerance=None,
                orientation_tolerance=None,
            )
            ik_ok[i] = bool(success)

            if not success:
                carb.log_warn(f"[plan] IK failed at waypoint {i}/{self.n_waypoints-1}. Stopping.")
                # hard stop , if IK fails
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

        # truncate to last successful waypoint
        last = int(np.where(ik_ok)[0][-1]) if np.any(ik_ok) else -1
        if last < 1:
            carb.log_warn("[plan] Not enough valid IK points to execute.")
            return None

        cart_wp = cart_wp[: last + 1]
        q_wp = q_wp[: last + 1]
        ik_ok = ik_ok[: last + 1]
        n = q_wp.shape[0]
        times = np.linspace(0.0, self.duration_s, n, dtype=np.float64)

        qd_wp = np.zeros_like(q_wp)
        for i in range(1, n):
            dt = times[i] - times[i - 1]
            if dt < 1e-9:
                qd_wp[i] = 0.0
            else:
                qd_wp[i] = (q_wp[i] - q_wp[i - 1]) / dt
                qd_wp[i] = clamp_qd(qd_wp[i], self.max_qd)
        qd_wp[0] = qd_wp[1].copy()

        carb.log_info(f"[plan] OK. duration={self.duration_s:.2f}s, waypoints={n}, dof={dof}")
        return TrajectoryPlan(
            duration_s=self.duration_s,
            dt_physics=float(dt_physics),
            eef_frame=self.eef_frame,
            cart_waypoints=cart_wp,
            times=times,
            q_waypoints=q_wp,
            qd_waypoints=qd_wp,
            ik_success=ik_ok,
        )

# EXECUTOR
class TrajectoryExecutor:
    """
    Applies the plan over time using physics dt.
    Logs:
      - plan details (once)
      - applied waypoint index
      - achieved EE pose (optional)
    """
    def __init__(self, robot: SingleArticulation, ik: ArticulationKinematicsSolver):
        self.robot = robot
        self.ik = ik

        self.plan: Optional[TrajectoryPlan] = None
        self.t = 0.0
        self.active = False

        self._printed_plan = False
        self.step_counter = 0

    def load_plan(self, plan: TrajectoryPlan):
        self.plan = plan
        self.t = 0.0
        self.active = True
        self._printed_plan = False

    def _sample(self, t: float) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Piecewise-linear sampling on joint positions.
        Returns (q, qd, seg_index).
        """
        p = self.plan
        assert p is not None

        times = p.times
        q_wp = p.q_waypoints
        qd_wp = p.qd_waypoints

        if t <= times[0]:
            return q_wp[0].copy(), qd_wp[0].copy(), 0
        if t >= times[-1]:
            return q_wp[-1].copy(), qd_wp[-1].copy(), len(times) - 1

        i = int(np.searchsorted(times, t) - 1)
        t0, t1 = times[i], times[i + 1]
        q0, q1 = q_wp[i], q_wp[i + 1]

        dt = t1 - t0
        if dt < 1e-9:
            return q0.copy(), np.zeros_like(q0), i

        a = (t - t0) / dt
        q = (1 - a) * q0 + a * q1
        qd = qd_wp[i].copy()
        return q, qd, i

    def _print_plan_once(self):
        if self.plan is None or self._printed_plan:
            return

        p = self.plan
        carb.log_info("========== TRAJECTORY PLAN ==========")
        carb.log_info(f"EEF frame: {p.eef_frame}")
        carb.log_info(f"Duration: {p.duration_s:.3f} s")
        carb.log_info(f"Physics dt: {p.dt_physics:.6f} s")
        carb.log_info(f"Waypoints: {len(p.times)}")
        carb.log_info("---- Cartesian waypoints (x,y,z) ----")
        for i, xyz in enumerate(p.cart_waypoints):
            carb.log_info(f"  wp[{i:02d}] = {xyz}")
        carb.log_info("---- Joint waypoints (rad) ----")
        for i, q in enumerate(p.q_waypoints):
            carb.log_info(f"  q[{i:02d}] = {q}")
        carb.log_info("---- IK success flags ----")
        carb.log_info(str(p.ik_success))
        carb.log_info("=====================================")

        self._printed_plan = True

    def step(self, dt: float):
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            return

        self.step_counter += 1

        if not self.active or self.plan is None:
            if self.step_counter % PRINT_EVERY_STEPS == 0:
                carb.log_info(f"[exec] idle (step={self.step_counter})")
            return

        if PRINT_PLAN_ONCE:
            self._print_plan_once()

        q, qd, seg = self._sample(self.t)

        action = ArticulationAction(
            joint_positions=q,
            joint_velocities=qd,
            joint_indices=list(range(len(q))),
        )
        self.robot.apply_action(action)

        if LOG_EE_TRACKING and (self.step_counter % PRINT_EVERY_STEPS == 0):
            ee_pos, _ = self.ik.compute_end_effector_pose(position_only=True)
            carb.log_info(f"[exec] step={self.step_counter} t={self.t:.3f}s seg={seg} EE={np.asarray(ee_pos)}")

        self.t += dt
        if self.t >= float(self.plan.times[-1]):
            self.active = False
            carb.log_info("[exec] Trajectory finished.")

#MAIN
world = None
robot = None
lula_solver = None
ik = None
planner = None
executor = None
requested_target_xyz = None
need_new_plan = False

def set_target_xyz(x: float, y: float, z: float):
    global requested_target_xyz, need_new_plan
    requested_target_xyz = np.array([x, y, z], dtype=np.float64)
    need_new_plan = True
    carb.log_info(f"[set_target_xyz] requested target = {requested_target_xyz}")

def on_physics_step(dt: float):
    global need_new_plan, requested_target_xyz
    global planner, executor

    if executor is None or planner is None:
        return

    # If a new target is requested: plan once, then execute
    if need_new_plan and requested_target_xyz is not None:
        need_new_plan = False

        plan = planner.plan(requested_target_xyz, dt_physics=dt)
        if plan is None:
            carb.log_warn("[main] Planning failed. No execution.")
        else:
            executor.load_plan(plan)

    # Always step executor
    executor.step(dt)

# Main Simulation 
async def main():
    global world, robot, lula_solver, ik, planner, executor

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

    # Kinematics
    kcfg = interface_config_loader.load_supported_lula_kinematics_solver_config("UR5e")
    lula_solver = LulaKinematicsSolver(**kcfg)

    frames = lula_solver.get_all_frame_names()
    carb.log_info(f"[init] available frames (count={len(frames)})")
    carb.log_info(f"[init] EEF '{EEF_FRAME_NAME}' exists? {EEF_FRAME_NAME in frames}")
    if EEF_FRAME_NAME not in frames:
        carb.log_warn("[init] Your EEF_FRAME_NAME is not in Lula frames. Fix this first.")
        # still continue, but IK will likely fail

    ik = ArticulationKinematicsSolver(robot, lula_solver, EEF_FRAME_NAME)
    planner = TrajectoryPlanner(robot=robot, lula_solver=lula_solver, ik=ik, eef_frame=EEF_FRAME_NAME, duration_s=TRAJ_DURATION_S, n_waypoints=N_WAYPOINTS, max_qd=MAX_QD_RAD_S,)
    executor = TrajectoryExecutor(robot=robot, ik=ik)

    world.add_physics_callback("refactored_cart_traj", on_physics_step)
    await world.play_async()

    set_target_xyz(0.75, 0.2, 0.5)


asyncio.ensure_future(main())

