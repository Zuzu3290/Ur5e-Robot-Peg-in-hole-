# UR5E Robot Simulation – Artificial Scripts Overview

This repository contains the simulation environment for the **UR5E robot** and several **artificial scripts** designed to emulate distinct aspects of the robot's dynamic behavior. Each script provides unique control or motion capabilities within the simulation.  

## Available Scripts

### 1. `Air_controller.py`
- **Purpose:** Demonstrates cube manipulation in the simulation.  
- **Functionality:** Uses **inverse kinematics** to align the robot’s end-effector with the position of a cube. This script focuses on presenting the cube and reinforcing positional matching during the simulation.  


### 2. `Articulat_standalone.py`
- **Purpose:** Provides fundamental control logic for the robot.  Executes pre-defined robot joint motions.  
- **Functionality:** Serves as a core controller template for integration with other motion and trajectory scripts. Emulates dynamic robot behavior using **pre-set joint configurations**.  
- **Notes:** This script does **not fully reflect constrained robot dynamics**, leading to rapid, “jumping” motions. It is under development and evaluation for smoother and more accurate robot dynamics.  

---

### 3. `EEF_controller.py`
- **Purpose:** Advanced Cartesian-space control.  
- **Functionality:**  
  - Sets a target in Cartesian space.  
  - Evaluates the robot’s current pose.  
  - Constructs trajectory waypoints and applies **inverse kinematics** to each joint to execute the motion.  
- **Notes:** Currently the most advanced control script in the repo. Under evaluation for potential integration with quadruped applications.  

---

### 4. `motion_path.py`
- **Purpose:** Modular motion control extension.  
- **Functionality:** Smoothly moves the robot from the current path to the target path.  
- **Notes:** Designed to **modularize robot motion**. Unlike `articulate_standalone.py`, it avoids rapid jumps and ensures smoother transitions. In the future, it may be integrated into `articulate_standalone.py`.  

---

## Summary

These scripts collectively demonstrate **robot simulation behavior** under various motion and control strategies. They allow exploration of:  
- Inverse kinematics-based motion  
- Joint-level dynamic manipulation  
- Trajectory planning and waypoint execution  
- Smooth vs. rapid motion behavior  

This modular structure provides a flexible framework for experimenting with **robot dynamics, control strategies, and trajectory optimization** in simulation.  

