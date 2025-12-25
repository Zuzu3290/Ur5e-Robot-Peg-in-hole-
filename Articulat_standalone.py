import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.stage import create_new_stage_async
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.world import World
from pxr import UsdPhysics
import asyncio

def deg2rad(joint_deg):
    return np.deg2rad(np.array(joint_deg, dtype=np.float32))

def current_positions(robot_view):
    positions = robot_view.get_joint_positions()
    print(f"Current joint positions: {positions}")
    return positions

async def action(robot_view, joint_positions, articulation_controller):

    current_positions(robot_view)
    action = ArticulationAction(joint_positions , joint_velocities=np.array([0.02] * 6), joint_indices=[0, 1, 2, 3, 4, 5])
    articulation_controller.apply_action(action)

    await asyncio.sleep(5.0)

    positions = robot_view.get_joint_positions()
    print(f"Current joint positions: {positions}")

async def joint_sensor(robot_view, prim_path):

    # Map joint names to link indices
    joint_link_id = dict()
    for joint_name in robot_view._articulation_view.joint_names:
        if joint_name == "base_link":
            continue  # Skip the base link
        try:
            joint_path = "/World/ur5e/joints/" + joint_name
            joint = UsdPhysics.Joint.Get(stage, joint_path)
            body_1_path = joint.GetBody1Rel().GetTargets()[0]
            body_1_name = stage.GetPrimAtPath(body_1_path).GetName()
            child_link_index = robot_view._articulation_view.get_link_index(body_1_name)
            joint_link_id[joint_name] = child_link_index
        except Exception as e:
            print(f"Skipping joint '{joint_name}' due to error: {e}")

    print("Joint link IDs:", joint_link_id)

# Access sensor data for specific UR5e joints
    for joint in ['shoulder_pan_joint', 'elbow_joint', 'wrist_3_joint']:
        idx = joint_link_id[joint]
        print(f"{joint} - Force: {sensor_joint_forces[idx]}, Effort: {sensor_actuation_efforts[idx]}")

async def articulation_controller():
    if World.instance():
        World.instance().clear_instance()
    world = World()
    await world.initialize_simulation_context_async()
    world.scene.add_default_ground_plane()

    # Load the robot USD file
    usd_path = r"C:\Users\zuhai\Desktop\IRP\ur5e_robot_calibrated\In_works.usd"
    prim_path = "/World/ur5e"
    hole_prim_path = "/World/ur5e_hole"
    add_reference_to_stage(usd_path, prim_path)

    # Create Articulation view for the robot
    robot_view = Articulation(prim_paths_expr=prim_path, name="UR5E_Robot")
    my_world.scene.add(robot_view)   # Reduntant line, already added below
    # Create and initialize the articulation controller with the articulation view
    articulation_controller = ArticulationController()
    articulation_controller.initialize(robot_view)

    # Run simulation
    await world.play_async()

    # Get current joint positions
    current_positions(robot_view)

    # articulation points 
    INITIAL = deg2rad([0, -41.4, 39.0, -90, -85.3, 147])
    ACT_1   = deg2rad([0, -25.6, 32.9, -90, -85.3, 147])
    ACT_2   = deg2rad([-1.9, -27.1, 36.6, -90, -85.3, 147])
    ACT_3   = deg2rad([-1.9, -24.8, 36.6, -90, -85.3, 147])

    # Get current joint positions
    await action(robot_view, INITIAL, articulation_controller)  # Run for 5 seconds to reach target positions

    await action(robot_view, ACT_1, articulation_controller)    # Run for 5 seconds to reach target positions

    await joint_sensor(robot_view, prim_path)

    await action(robot_view, INITIAL, articulation_controller)    # Run for 5 seconds to reach target positions 

    await action(robot_view, ACT_2, articulation_controller)    # Run for 5 seconds to reach target positions

    await joint_sensor(robot_view, prim_path)

    await action(robot_view, INITIAL, articulation_controller)    # Run for 5 seconds to reach target positions

    await action(robot_view, ACT_3, articulation_controller)    # Run for 5 seconds to reach target positions
    
    await joint_sensor(robot_view, prim_path)
    
    await action(robot_view, INITIAL, articulation_controller)    # Run for 5 seconds to reach target positions

    world.pause()

# Run the example
asyncio.ensure_future(articulation_controller())