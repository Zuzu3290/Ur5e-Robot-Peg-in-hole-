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
from isaacsim.core.api.world import World
from pxr import *
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

async def joint_sensor(arti_view, joint_link_id, positions):    

    joints_name = [
    "shoulder_pan_joint",   # index 0
    "shoulder_lift_joint",  # index 1
    "elbow_joint",          # index 2
    "wrist_1_joint",        # index 3
    "wrist_2_joint",        # index 4
    "wrist_3_joint"         # index 5
    ]

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
    # Get current joint positions
    current_phase = "INITIAL"
    await action(robot_view, INITIAL, articulation_controller, arti_view, joint_link_id, contact_sensor)  # Run for 5 seconds to reach target positions

    current_phase = "ACT_1"
    await action(robot_view, ACT_1, articulation_controller, arti_view, joint_link_id, contact_sensor)    # Run for 5 seconds to reach target positions

    current_phase = "INITIAL"
    await action(robot_view, INITIAL, articulation_controller, arti_view, joint_link_id, contact_sensor)    # Run for 5 seconds to reach target positions 

    current_phase = "ACT_2"
    await action(robot_view, ACT_2, articulation_controller, arti_view, joint_link_id, contact_sensor)    # Run for 5 seconds to reach target positions

    current_phase = "INITIAL"
    await action(robot_view, INITIAL, articulation_controller, arti_view, joint_link_id, contact_sensor)    # Run for 5 seconds to reach target positions

    current_phase = "ACT_3"
    await action(robot_view, ACT_3, articulation_controller, arti_view, joint_link_id, contact_sensor)    # Run for 5 seconds to reach target positions

    current_phase = "INITIAL"
    await action(robot_view, INITIAL, articulation_controller, arti_view, joint_link_id, contact_sensor)    # Run for 5 seconds to reach target positions
    
    df = pd.DataFrame(joint_log)
    df.to_csv(output_dir, sep='|', index=False, mode='w', header=True, columns=[ "joint_name", "position_rad", "force_N", "torque_Nm", "timestamp","phase"])
    
    df  = pd.DataFrame(contact_sensor_log)
    df.to_csv(contact_sensor_path, sep='|', index=False, mode='w', header=True, columns=[ "contact_force_N", "timestamp","Step_number","phase"])

    world.pause()

# Run the example
asyncio.ensure_future(articulation_controller())