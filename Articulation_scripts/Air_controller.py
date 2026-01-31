import asyncio
import numpy as np
import omni
import carb
from pxr import UsdGeom, Gf

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage

from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)

# ----------------------------
# SETTINGS
# ----------------------------
USD_PATH = r"C:\Users\zuhai\Desktop\IRP\ur5e_robot_calibrated\ur5e_rework.usd"
ROBOT_PRIM = "/World/ur5e"

# Kinematics end-effector frame name (must be in lula_solver.get_all_frame_names())
EEF_FRAME_NAME = "wrist_3_link"

# USD prim path where we attach a visual marker for EEF (just for visualization)
EEF_VIS_PRIM_PATH = "/World/ur5e/wrist_3_link"
EE_MARKER_PRIM = EEF_VIS_PRIM_PATH + "/ee_marker_cube"

# World target cube you DRAG
TARGET_CUBE_PRIM = "/World/target_cube"

# Keep wrist_3_link at least this far from cube center (meters)
STANDOFF_M = 0.08

PRINT_EVERY = 60
WARN_EVERY = 30

# ----------------------------
# GLOBALS
# ----------------------------
world = None
robot = None
stage = None
lula_solver = None
ik = None

target_cube_prim = None
ee_marker_prim = None

step_count = 0
last_good_action = None


def to_np(x, dtype=np.float64):
    """Convert torch Tensor / list to numpy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(dtype)
    return np.asarray(x, dtype=dtype)


def create_colored_cube(stage, prim_path: str, size=0.06, color=(1.0, 0.65, 0.65), pos=(0.0, 0.0, 0.0)):
    """Create/reuse a USD cube with displayColor and a translate op."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.CreateSizeAttr(float(size))
        prim = stage.GetPrimAtPath(prim_path)
    else:
        cube = UsdGeom.Cube(prim)
        cube.CreateSizeAttr(float(size))

    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr([1.0])

    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    t_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t_op = op
            break
    if t_op is None:
        t_op = xform.AddTranslateOp()

    t_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    return prim


def get_world_translation(prim):
    """Live world position from USD (updates immediately when you drag in viewport)."""
    mat = omni.usd.get_world_transform_matrix(prim)
    t = mat.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=np.float64)


def compute_standoff_target(ee_pos: np.ndarray, cube_pos: np.ndarray, standoff_m: float) -> np.ndarray:
    """
    Ensure target point is at least standoff_m away from cube center along cube->EE direction.
    This makes wrist_3_link stop at a safe distance instead of touching/penetrating.
    """
    v = ee_pos - cube_pos
    dist = float(np.linalg.norm(v))
    if dist < 1e-6:
        return cube_pos + np.array([0.0, 0.0, standoff_m], dtype=np.float64)
    return cube_pos + (v / dist) * standoff_m


def on_physics_step(dt: float):
    global step_count, last_good_action
    global robot, lula_solver, ik, target_cube_prim, ee_marker_prim

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        return

    step_count += 1

    # 1) Cartesian target pose: read dragged cube world position
    cube_pos = get_world_translation(target_cube_prim)

    # 2) Current EEF position (wrist_3_link visual marker)
    ee_pos = get_world_translation(ee_marker_prim)

    # 3) Enforce standoff distance (safe Cartesian target position)
    target_pos = compute_standoff_target(ee_pos, cube_pos, STANDOFF_M)

    # 4) Update Lula base pose (numpy only; backend is torch)
    b_pos, b_quat = robot.get_world_pose()
    lula_solver.set_robot_base_pose(to_np(b_pos), to_np(b_quat))

    # 5) Compute IK -> ArticulationAction
    action, success = ik.compute_inverse_kinematics(
        target_position=target_pos,
        target_orientation=None,  # position-only tracking
        position_tolerance=None,
        orientation_tolerance=None,
    )

    # 6) Apply action (and handle out-of-range)
    if success:
        robot.apply_action(action)
        last_good_action = action
    else:
        # Hold last valid posture instead of "nullifying"
        if last_good_action is not None:
            robot.apply_action(last_good_action)
        if step_count % WARN_EVERY == 0:
            carb.log_warn("IK failed (likely out of reachable range). Holding last valid posture.")

    if step_count % PRINT_EVERY == 0:
        carb.log_info(f"[{step_count}] IK success={success} cube={cube_pos} target={target_pos} standoff={STANDOFF_M}m")


async def main():
    global world, robot, stage, lula_solver, ik, target_cube_prim, ee_marker_prim

    if World.instance():
        World.instance().clear_instance()

    world = World(stage_units_in_meters=1.0, backend="torch")
    await world.initialize_simulation_context_async()
    await omni.kit.app.get_app().next_update_async()
    world.scene.add_default_ground_plane()

    # Load robot
    add_reference_to_stage(USD_PATH, ROBOT_PRIM)
    await omni.kit.app.get_app().next_update_async()

    robot = SingleArticulation(prim_path=ROBOT_PRIM, name="ur5e")
    world.scene.add(robot)

    await world.reset_async(soft=False)
    stage = omni.usd.get_context().get_stage()

    # Visual EEF marker (blue) attached under wrist_3_link
    ee_marker_prim = create_colored_cube(
        stage,
        EE_MARKER_PRIM,
        size=0.035,
        color=(0.6, 0.8, 1.0),
        pos=(0.0, 0.0, 0.0),
    )
    await omni.kit.app.get_app().next_update_async()

    # Target cube (light red) in world (drag this)
    target_cube_prim = create_colored_cube(
        stage,
        TARGET_CUBE_PRIM,
        size=0.06,
        color=(1.0, 0.65, 0.65),
        pos=(0.45, 0.0, 0.35),
    )
    await omni.kit.app.get_app().next_update_async()

    # Lula kinematics
    kcfg = interface_config_loader.load_supported_lula_kinematics_solver_config("UR5e")
    lula_solver = LulaKinematicsSolver(**kcfg)

    frames = lula_solver.get_all_frame_names()
    carb.log_info(f"EEF '{EEF_FRAME_NAME}' in frames? {EEF_FRAME_NAME in frames}")

    # IK wrapper: returns ArticulationAction directly
    ik = ArticulationKinematicsSolver(robot, lula_solver, EEF_FRAME_NAME)

    # Live update loop
    world.add_physics_callback("cartesian_target_ik_servo", on_physics_step)

    # Start simulation
    await world.play_async()


asyncio.ensure_future(main())



#Issue presists that we need a loop induction. The program doesnt compute change in enf_effector coordinate during runtimee. 
#We need to add a loop that constantly checks for changes in target coordinates and re-plans the trajectory accordingly.async def monitor_target_changes():
#We need to print the necessary logs that describt eh induction of the manipulating actiosn applied during runtime following the waypoints
#We need to add the contact sernsor as well as the force api that collects the data related to the forces applied on the end-effector during manipulation tasks.
#We need to save this daat and plot the graph accordingly 
#We need to recontruct the world stage and evaulete the changes and the dynamics of the envrionement such that we can induct proper cartesian coordinates and waypoints. 
#Proper incooperation of the IK applictaion and the waypoitns generated. 
#Work with the Articulationot single articulation api to evaulete the changes in the joint positions and velocities during runtime.
#Optimally it would be better to refactor themotion inducted by the robto under the coordiantes, hence enfrocing restrains and proper manipulation tasks.
