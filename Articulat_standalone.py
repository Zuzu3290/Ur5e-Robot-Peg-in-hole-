import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.stage import create_new_stage_async
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.prims import Articulation
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.world import World
from pxr import UsdPhysics
import asyncio
import omni
import carb
from pxr import UsdGeom
from isaacsim.sensors.physics import _sensor
import matplotlib.pyplot as plt

#ur5e joint names
joints_name = [
    "shoulder_pan_joint",   # index 0
    "shoulder_lift_joint",  # index 1
    "elbow_joint",          # index 2
    "wrist_1_joint",        # index 3
    "wrist_2_joint",        # index 4
    "wrist_3_joint"         # index 5
    ]

CONTACT_SENSOR_PATH = "/World/ur5e/wrist_3_link/ur5e_peg/Contact_Sensor"
PRINT_EVERY_N_STEPS = 10

def vector_magnitude(v):
    
    x = v[0]
    y = v[1]
    z = v[2]
    return np.sqrt(x*x + y*y + z*z)

def plot_joint_forces(joint_names, joint_forces):
  
    joint_indices = np.arange(len(joint_names))

    plt.figure(figsize=(10, 5))
    plt.bar(joint_indices, joint_forces)
    plt.xticks(joint_indices, joint_names, rotation=45)
    plt.ylabel("Force Magnitude (N)")
    plt.xlabel("Joint Position")
    plt.title("Joint Force Magnitudes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_joint_torques(joint_names, joint_torques):
   
    joint_indices = np.arange(len(joint_names))

    plt.figure(figsize=(10, 5))
    plt.plot(joint_indices, joint_torques, marker="o")
    plt.xticks(joint_indices, joint_names, rotation=45)
    plt.ylabel("Torque Magnitude (Nm)")
    plt.xlabel("Joint Position")
    plt.title("Joint Torque Magnitudes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def deg2rad(joint_deg):
    return np.deg2rad(np.array(joint_deg, dtype=np.float32))

def current_positions(robot_view,joints_name):
    positions = robot_view.get_joint_positions()
    positions = positions.detach().numpy().flatten()

    for i in range(len(joints_name)):
        print(f" {joints_name[i]}: {positions[i]}")

async def action(robot_view, joint_positions, articulation_controller,arti_view, joint_link_id, contact_sensor_interface):
    joints_name = [
    "shoulder_pan_joint",   # index 0
    "shoulder_lift_joint",  # index 1
    "elbow_joint",          # index 2
    "wrist_1_joint",        # index 3
    "wrist_2_joint",        # index 4
    "wrist_3_joint"         # index 5
    ]

    current_positions(robot_view,joints_name)
    action = ArticulationAction(joint_positions , joint_velocities=np.array([0.2] * 6), joint_indices=[0, 1, 2, 3, 4, 5])
    articulation_controller.apply_action(action)

    positions = robot_view.get_joint_positions()
    positions = positions.detach().numpy().flatten()

    for i in range(len(joints_name)):
        print(f" {joints_name[i]}: {positions[i]}")
  
    #Pause this async function until the next Kit (Isaac Sim) update frame has completed.
    await omni.kit.app.get_app().next_update_async()
    
    # # Contact sensor application
    # reading = contact_sensor_interface.get_sensor_reading(CONTACT_SENSOR_PATH)

    # if reading.is_valid:
    #     force_n = float(reading.value)
    #     carb.log_info(f"[Peg Contact] Force = {force_n:.4f} N")

    await joint_sensor(arti_view, joint_link_id)

    await omni.kit.app.get_app().next_update_async()

    await asyncio.sleep(5.0)

async def joint_sensor(arti_view, joint_link_id):

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

        print(f"Joint: {joint_name}")
        print(f"  Force  [Fx, Fy, Fz]: {joint_force[:3]}")
        print(f"  Torque [Tx, Ty, Tz]: {joint_force[3:]}")
        print(f"  Actuation Effort: {joint_effort}")
        print("-" * 40)

    for joint_index, joint_name in enumerate(joints_name):

        joint_wrench = sensor_joint_forces[env_id, joint_index]

        force_vec = joint_wrench[:3]   # Fx, Fy, Fz
        torque_vec = joint_wrench[3:]  # Tx, Ty, Tz

        force_mag = vector_magnitude(force_vec)
        torque_mag = vector_magnitude(torque_vec)

        force_magnitudes.append(force_mag)
        torque_magnitudes.append(torque_mag)
        
        print(f"Joint: {joint_name}")
        print(f"  Force  [Fx, Fy, Fz]: {force_vec} | Magnitude: {force_mag:.4f} N")
        print(f"  Torque [Tx, Ty, Tz]: {torque_vec} | Magnitude: {torque_mag:.4f} Nm")
        print("-" * 40)
    
    # Plot results
    plot_joint_forces(joints_name, force_magnitudes)
    plot_joint_torques(joints_name, torque_magnitudes)

async def articulation_controller():
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
    joints_name = [
    "shoulder_pan_joint",   # index 0
    "shoulder_lift_joint",  # index 1
    "elbow_joint",          # index 2
    "wrist_1_joint",        # index 3
    "wrist_2_joint",        # index 4
    "wrist_3_joint"         # index 5
    ]
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

    #the upper code is used to modulate if simulation is being operated manually
    contact_sensor_interface = _sensor.acquire_contact_sensor_interface()

    # Get current joint positions
    current_positions(robot_view, joints_name)

    # articulation points 
    INITIAL = deg2rad([-4, -40.3, 37.9, -90, -90, 0.0])
    ACT_1   = deg2rad([-4, -32, 38.5, -90, -90, 0.0])
    ACT_2   = deg2rad([-4.3, -30.4, 38.5, -90, -92.4, 0.0])
    ACT_3   = deg2rad([-4.3, -28.2, 38.5, -90, -92.4, 0.0])

    # Get current joint positions
    await action(robot_view, INITIAL, articulation_controller, arti_view, joint_link_id, contact_sensor_interface)  # Run for 5 seconds to reach target positions

    await action(robot_view, ACT_1, articulation_controller, arti_view, joint_link_id, contact_sensor_interface)    # Run for 5 seconds to reach target positions

    await action(robot_view, INITIAL, articulation_controller, arti_view, joint_link_id, contact_sensor_interface)    # Run for 5 seconds to reach target positions 

    await action(robot_view, ACT_2, articulation_controller, arti_view, joint_link_id, contact_sensor_interface)    # Run for 5 seconds to reach target positions

    await action(robot_view, INITIAL, articulation_controller, arti_view, joint_link_id, contact_sensor_interface)    # Run for 5 seconds to reach target positions

    await action(robot_view, ACT_3, articulation_controller, arti_view, joint_link_id, contact_sensor_interface)    # Run for 5 seconds to reach target positions

    await action(robot_view, INITIAL, articulation_controller, arti_view, joint_link_id, contact_sensor_interface)    # Run for 5 seconds to reach target positions

    world.pause()

# Run the example
asyncio.ensure_future(articulation_controller())
