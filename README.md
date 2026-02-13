# UR5E Robot Peg-in-Hole Simulation

### Overview 
This repository provides a simulation environment for studying the **peg-in-hole task using the UR5E robotic arm within NVIDIA Isaac Sim**. The primary objective is to analyze and evaluate the robot's behavior during peg insertion tasks, considering various simulation conditions and control algorithms.

Isaac Sim leverages NVIDIA’s powerful simulation framework, providing realistic physics and sensor data, which makes it an ideal tool for developing, testing, and refining robotics algorithms in a virtual environment.

---
### Objectives 
* Study Robot Behavior
* Test Control Algorithms
* Visualize Forces
* Simulate Articulated Joints:
---
### Acheivemnets 
* Articulated Joints Simulation
* Force Visualization
* Descriptive Media
* Easy Exploration
---
### Setup Process 
Prerequisites: 
* [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/quick-install.html) The hyperlink would automatically install the most stable version. At the current moment the most stabel version is 5.1. 6.0 is under development and only installable via the github source.
* Python
---
> ⚠️ Caution:
> 
> During the simulation, it is recommended to copy and paste the code instead of importing the scripts directly. Currently, all scripts are designed for direct import except for the UR5E script.
> 
> The reason for this recommendation is that the simulation environment automatically attempts to run the simulation whenever changes are made. This can result in glitches and may even cause the app to crash during debugging or improvements. 
---
## Steps
1) Install Isaac sim 
2) Follow the documentation to run the App
3) Import the ur5e.usd file
4) Click on Window -> Script Editor
5) Click on File -> Import
---

## Repository Structure
### Articulation Scripts
> **[Air_contoller.py](./Articulation_scripts\Air_controller.py)** Simulates the robot using a floating cube mesh in mid-air. The end effector follows the path dictated by the cube’s movement. 
> 
> **[Articulat_standalone.py](./Articulation_scripts\Articulat_standalone.py)** Simulates the robot’s motion based on predefined performance parameters for each joint of the UR5E.
> 
> **[EFF_controller.py](./Articulation_scripts\EEF_controller.py)** Moves the robot’s end effector to target coordinates in the real world using the custom-built API: set_target(x, y, z).
> 
> **[UR5e.py](./Articulation_scripts\UR5e.py)** Designing to simulate the robot trajectories by parsing parameters in Cartesian space via the set_target() API.
> 



