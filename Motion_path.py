import numpy as np
import asyncio
import carb
import os
import pandas as pd

from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage_async, get_current_stage
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.world import World
from isaacsim.sensors.physics import _sensor

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
robot_usd_path = r"C:\Users\zuhai\Desktop\IRP\UR robot file\ur5e_edit.usd"
robot_prim_path = "/World/ur5e"

joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",    
    "wrist_2_joint",
    "wrist_3_joint"
]

output_dir = r"C:\Users\zuhai\Desktop\IRP\ur5e_robot_calibrated\joint_data.csv"
contact_sensor_path = r"C:\Users\zuhai\Desktop\IRP\ur5e_robot_calibrated\contact_sensor_data.csv"
Contact_Sensor_path = "/World/ur5e/wrist_3_link/ur5e_peg/Contact_Sensor"

joint_log = []
contact_sensor_log = []
current_phase = ""

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def deg2rad(joint_deg):
    return np.deg2rad(np.array(joint_deg, dtype=np.float32))

def vector_magnitude(v):
    return np.sqrt(np.sum(np.array(v)**2))

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
        "Step_number": step,
        "phase": current_phase
    })

# ------------------------------------------------------------
# Procedural trajectory execution
# ------------------------------------------------------------
async def move_to_target(robot_view, articulation_controller, target_positions, duration=5.0):
    # Ensure target positions are NumPy
    if hasattr(target_positions, "detach"):
        q_goal = target_positions.detach().cpu().numpy().flatten()
    else:
        q_goal = np.array(target_positions, dtype=np.float32).flatten()

    physics_dt = 1/60.0
    steps = int(duration / physics_dt)

    for step in range(steps + 1):
        alpha = step / steps

        # Get current joint positions as NumPy
        q_start = robot_view.get_joint_positions()
        if hasattr(q_start, "detach"):
            q_start = q_start.detach().cpu().numpy().flatten()
        else:
            q_start = np.array(q_start, dtype=np.float32).flatten()

        q = (1 - alpha) * q_start + alpha * q_goal
        qd = (q_goal - q_start) / duration

        action = ArticulationAction(
            joint_positions=q,
            joint_velocities=qd,
            joint_indices=[0, 1, 2, 3, 4, 5]
        )
        articulation_controller.apply_action(action)

        # Optional: contact sensor reading
        contact_sensor = _sensor.acquire_contact_sensor_interface()
        reading = contact_sensor.get_sensor_reading(Contact_Sensor_path)
        if reading.is_valid and reading.value > 0:
            print(f"[{current_phase}] Step {step}: Contact force = {reading.value:.4f} N")

        await omni.kit.app.get_app().next_update_async()

# ------------------------------------------------------------
# Main routine
# ------------------------------------------------------------
async def run_robot_trajectory():
    global current_phase

    # --- Stage & World ---
    await create_new_stage_async()
    stage = get_current_stage()

    if not os.path.isfile(robot_usd_path):
        carb.log_error(f"USD file not found: {robot_usd_path}")
        return

    add_reference_to_stage(robot_usd_path, robot_prim_path)
    await omni.kit.app.get_app().next_update_async()

    robot_view = Articulation(prim_paths_expr=robot_prim_path, name="UR5E_Robot")
    articulation_controller = ArticulationController()
    articulation_controller.initialize(robot_view)

    world = World(stage_units_in_meters=1.0, backend="torch")
    await world.initialize_simulation_context_async()
    await omni.kit.app.get_app().next_update_async()
    world.scene.add_default_ground_plane()
    await world.reset_async(soft=False)
    await world.play_async()

    # Define joint targets
    INITIAL = deg2rad([-4, -40.3, 37.9, -90, -90, 0.0])
    ACT_1   = deg2rad([-2.8, -29.7, 36.1, -90, -90, 0.0])
    ACT_2   = deg2rad([-4.3, -30.4, 38.5, -90, -92.4, 0.0])
    ACT_3   = deg2rad([-4.3, -28.2, 38.5, -90, -92.4, 0.0])

    # Run phases
    for phase_name, target in [("ACT_1", ACT_1), ("INITIAL", INITIAL),
                               ("ACT_2", ACT_2), ("ACT_3", ACT_3), ("INITIAL", INITIAL)]:
        current_phase = phase_name
        await move_to_target(robot_view, articulation_controller, target, duration=5.0)

    # Save logs
    pd.DataFrame(joint_log).to_csv(output_dir, sep='|', index=False)
    pd.DataFrame(contact_sensor_log).to_csv(contact_sensor_path, sep='|', index=False)

    world.pause()

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
asyncio.ensure_future(run_robot_trajectory())
carb.log_info("Robot trajectory execution started.")