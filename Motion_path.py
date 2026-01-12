import numpy as np
import pandas as pd
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.stage import create_new_stage_async
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.prims import Articulation
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import PathPlannerVisualizer
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.robot_motion.motion_generation import interface_config_loader
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.api.world import World
import isaacsim.core.api.objects # This library provides can be used to provide the location/position of objects within the stage 
from pxr import *
from pathlib import Path
import asyncio
import omni
import carb
from isaacsim.sensors.physics import _sensor


Contact_Sensor_path = "/World/ur5e/wrist_3_link/ur5e_peg/Contact_Sensor"
output_dir = r"C:\Users\zuhai\Desktop\IRP\ur5e_robot_calibrated\joint_data.csv"
contact_sensor_path = r"C:\Users\zuhai\Desktop\IRP\ur5e_robot_calibrated\contact_sensor_data.csv"
joint_log = [] 
contact_sensor_log = []
#ur5e joint names
joints_name = [
    "shoulder_pan_joint",   # index 0
    "shoulder_lift_joint",  # index 1
    "elbow_joint",          # index 2
    "wrist_1_joint",        # index 3
    "wrist_2_joint",        # index 4
    "wrist_3_joint"         # index 5
    ]

def halt_simulation():
    halt_sim = False
    global watch_joint
    global active_joint
    watch_joint =RRT.get_watched_joints() # Observing the joints
    active_joint = RRT.get_active_joints() # Joints being controlled 

    if watch_joint == active_joint:
        carb.log_info("Watched joints and Active joints are the same.")
    else:
        carb.log_info("Watched joints and Active joints are different.")
        halt_sim = True

    if halt_sim==True:
        carb.log_info("Halting the simulation due to configuration mismatch.")
        app = omni.kit.app.get_app()
        app.stop()
         
def log_joint_data(joint_name, position, force, torque):
    joint_log.append({
        "joint_name": joint_name,
        "position_rad": float(position),
        "force_N": float(force),
        "torque_Nm": float(torque),
        "timestamp": omni.timeline.get_timeline_interface().get_current_time(),
        "phase": current_phase
        })

def log_contact_data(contact_force, step):
    contact_sensor_log.append({
        "contact_force_N": float(contact_force),
        "timestamp": omni.timeline.get_timeline_interface().get_current_time(),
        "Step_number" : step,
        "phase": current_phase
    })

def vector_magnitude(v):
    
    x = v[0]
    y = v[1]
    z = v[2]
    return np.sqrt(x*x + y*y + z*z)

def deg2rad(joint_deg):
    return np.deg2rad(np.array(joint_deg, dtype=np.float32))

def current_positions(robot_view,joints_name):
    positions = robot_view.get_joint_positions()
    positions = positions.detach().numpy().flatten()

    for i in range(len(joints_name)):
        log_joint_data( joints_name[i], positions[i], 0, 0)
        print(f" {joints_name[i]}: {positions[i]}")

async def action(robot_view, joint_positions, articulation_controller,arti_view, joint_link_id, contact_sensor):
    joints_name = [
    "shoulder_pan_joint",   # index 0
    "shoulder_lift_joint",  # index 1
    "elbow_joint",          # index 2
    "wrist_1_joint",        # index 3
    "wrist_2_joint",        # index 4
    "wrist_3_joint"         # index 5
    ]

    current_positions(robot_view,joints_name)
    action = ArticulationAction(joint_positions , joint_velocities=np.array([0.0,0.005,0.02,0.0,0.0,0.0]), joint_indices=[0, 1, 2, 3, 4, 5])
    articulation_controller.apply_action(action)
    
    for step in range(600):  # ~10 seconds
        await omni.kit.app.get_app().next_update_async()

        reading = contact_sensor.get_sensor_reading(Contact_Sensor_path)
        if reading.is_valid:
            if reading.value > 0:
                print(f"[{step}] Contact force: {reading.value:.4f} N")
                log_contact_data(reading.value, step)

    positions = robot_view.get_joint_positions()
    positions = positions.detach().numpy().flatten()
    
    await omni.kit.app.get_app().next_update_async()
    # Contact sensor application
    await joint_sensor(arti_view, joint_link_id, positions)

    await omni.kit.app.get_app().next_update_async()

    await asyncio.sleep(5.0)

async def Waypoint_action(robot_view, joint_positions, articulation_controller, arti_view, joint_link_id, contact_sensor):

    global joints_name

    # Log current joint positions
    current_positions(robot_view, joints_name)

    mg_extension_path = Path(get_extension_path_from_name("isaacsim.robot_motion.motion_generation"))
    rmp_config_dir = mg_extension_path / "motion_policy_configs"
    rrt_config_dir = mg_extension_path / "path_planner_configs"

    # Initialize an RRT object for planning
    robot_view._rrt = RRT(
        robot_description_path=str(rmp_config_dir / "franka/rmpflow/robot_descriptor.yaml"),
        urdf_path=str(rmp_config_dir / "franka/lula_franka_gen.urdf"),
        rrt_config_path=str(rrt_config_dir / "franka/rrt/franka_planner_config.yaml"),
        end_effector_frame_name="flange"
    )
    robot_view._rrt.set_max_iterations(5000)

    # Setup the PathPlannerVisualizer
    robot_view._path_planner_visualizer = PathPlannerVisualizer(robot_view._articulation, robot_view._rrt)

    # Replan if the target has moved
    current_target_translation, current_target_orientation = robot_view._target.get_world_pose()
    current_target_rotation = quats_to_rot_matrices(current_target_orientation)

    translation_distance = np.linalg.norm(robot_view._target_translation - current_target_translation)
    rotation_distance = rotational_distance_angle(current_target_rotation, robot_view._target_rotation)
    target_moved = translation_distance > 0.01 or rotation_distance > 0.01

    if (robot_view._frame_counter % 60 == 0 and target_moved):
        robot_view._rrt.set_end_effector_target(current_target_translation, current_target_orientation)
        robot_view._rrt.update_world()
        robot_view._plan = robot_view._path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=.01)

        robot_view._target_translation = current_target_translation
        robot_view._target_rotation = current_target_rotation

    # Execute the planned actions
    if robot_view._plan:
        for i, action_target in enumerate(robot_view._plan):
            action_cmd = ArticulationAction(action_target, joint_indices=[0,1,2,3,4,5])
            articulation_controller.apply_action(action_cmd)

            # If last waypoint, log contact for ~10 seconds
            if i == len(robot_view._plan) - 1:
                for step in range(600):
                    await omni.kit.app.get_app().next_update_async()
                    reading = contact_sensor.get_sensor_reading(Contact_Sensor_path)
                    if reading.is_valid and reading.value > 0:
                        print(f"[{step}] Contact force: {reading.value:.4f} N")
                        log_contact_data(reading.value, step)

                await asyncio.sleep(5.0)  # wait at last position

    # Log joint sensor data at final position
    positions = robot_view.get_joint_positions().detach().numpy().flatten()
    await joint_sensor(arti_view, joint_link_id, positions)

async def joint_sensor(arti_view, joint_link_id, positions):    

    global joints_name

    sensor_joint_forces = arti_view.get_measured_joint_forces()
    sensor_actuation_efforts = arti_view.get_measured_joint_efforts()
    
    env_id = 0  # single robot → always 0
    force_magnitudes = []
    torque_magnitudes = []

    print("\n=== Joint Force Report ===")

    for joint_index, joint_name in enumerate(joints_name):

        joint_force = sensor_joint_forces[env_id, joint_index]
        joint_effort = sensor_actuation_efforts[env_id, joint_index]
        
        force_vec = joint_force[:3]   # Fx, Fy, Fz
        torque_vec = joint_force[3:]  # Tx, Ty, Tz

        force_mag = vector_magnitude(force_vec)
        torque_mag = vector_magnitude(torque_vec)

        force_mag = float(force_mag)
        torque_mag = float(torque_mag)

        log_joint_data(joints_name[joint_index], positions[joint_index], force_mag, torque_mag)

        # force_magnitudes.append(force_mag)
        # torque_magnitudes.append(torque_mag)

        print(f"Joint: {joint_name}")
        print(f"  Force  [Fx, Fy, Fz]: {force_vec} | Magnitude: {force_mag:.4f} N")
        print(f"  Torque [Tx, Ty, Tz]: {torque_vec} | Magnitude: {torque_mag:.4f} Nm")
        print(f"  Actuation Effort: {joint_effort}")
        print("-" * 40)
   
async def articulation_controller(joints_name=joints_name):

    # Intilaize the simulation world
    if World.instance():
        World.instance().clear_instance()
    world = World(stage_units_in_meters=1.0, backend="torch")
    await world.initialize_simulation_context_async()
    await omni.kit.app.get_app().next_update_async()
    world.scene.add_default_ground_plane()
    
    # Load the robot USD file
    usd_path = r"C:\Users\zuhai\Desktop\IRP\ur5e_robot_calibrated\ur5e_rework.usd"
    prim_path = "/World/ur5e"
    hole_prim_path = "/World/ur5e_hole"
    PRINT_EVERY_N_STEPS = 10
    add_reference_to_stage(usd_path, prim_path)

    # Create Articulation view for the robot
    robot_view = Articulation(prim_paths_expr=prim_path, name="UR5E_Robot")
    # Create and initialize the articulation controller with the articulation view
    articulation_controller = ArticulationController()
    articulation_controller.initialize(robot_view)

    # Create Articution Object
    arti_view = Articulation("/World/ur5e/base_link")
    world.scene.add(arti_view)
    await omni.kit.app.get_app().next_update_async()
    #Fully reset the simulation world to its initial physical state and reinitialize all physics and articulation buffers
    await world.reset_async(soft=False)
    stage = get_current_stage()

    # Iterates through the joint names in the articulation, retrieves information about the joints and their associated links,
    # and creates a mapping between joint names and their corresponding link indices.
    
    joint_link_id = dict()
    for joint_name in arti_view.joint_names:
        joint_path = "/World/ur5e/joints/" + joint_name
        joint = UsdPhysics.Joint.Get(stage, joint_path)
        body_1_path = joint.GetBody1Rel().GetTargets()[0]
        body_1_name = stage.GetPrimAtPath(body_1_path).GetName()
        child_link_index = arti_view.get_link_index(body_1_name)
        joint_link_id[joint_name] = child_link_index
    print("joint link IDs", joint_link_id)

    # Run simulation
    await world.play_async()

    contact_sensor = _sensor.acquire_contact_sensor_interface()
    reading = contact_sensor.get_sensor_reading(Contact_Sensor_path)

    # articulation points 
    INITIAL = deg2rad([-4, -40.3, 37.9, -90, -90, 0.0])
    ACT_1   = deg2rad([-2.8, -29.7, 36.1, -90, -90.0, 0.0])
    ACT_2   = deg2rad([-4.3, -30.4, 38.5, -90, -92.4, 0.0])
    ACT_3   = deg2rad([-4.3, -28.2, 38.5, -90, -92.4, 0.0])

    global current_phase

    current_phase = "INITIAL"
    await action(robot_view, INITIAL, articulation_controller, arti_view, joint_link_id, contact_sensor)  # Run for 5 seconds to reach target positions

    current_phase = "ACT_1"
    await Waypoint_action(robot_view, ACT_1, articulation_controller, arti_view, joint_link_id, contact_sensor)
    halt_simulation()

    current_phase = "INITIAL"
    await action(robot_view, INITIAL, articulation_controller, arti_view, joint_link_id, contact_sensor)    # Run for 5 seconds to reach target positions 

    current_phase = "ACT_2"
    await Waypoint_action(robot_view, ACT_2, articulation_controller, arti_view, joint_link_id, contact_sensor)
    halt_simulation()

    current_phase = "INITIAL"
    await action(robot_view, INITIAL, articulation_controller, arti_view, joint_link_id, contact_sensor)    # Run for 5 seconds to reach target positions

    current_phase = "ACT_3"
    await Waypoint_action(robot_view, ACT_3, articulation_controller, arti_view, joint_link_id, contact_sensor)
    halt_simulation()

    current_phase = "INITIAL"
    await action(robot_view, INITIAL, articulation_controller, arti_view, joint_link_id, contact_sensor)    # Run for 5 seconds to reach target positions
    
    df = pd.DataFrame(joint_log)
    df.to_csv(output_dir, sep='|', index=False, mode='w', header=True, columns=[ "joint_name", "position_rad", "force_N", "torque_Nm", "timestamp","phase"])
    
    df  = pd.DataFrame(contact_sensor_log)
    df.to_csv(contact_sensor_path, sep='|', index=False, mode='w', header=True, columns=[ "contact_force_N", "timestamp","Step_number","phase"])

    world.pause()

# Run the example
asyncio.ensure_future(articulation_controller())