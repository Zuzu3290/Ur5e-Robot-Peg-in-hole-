import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("joint_data.csv")

joints_name = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]
#Force vs Position
plt.figure(figsize=(10, 6))

for joint in joints_name:
    joint_df = df[df["joint_name"] == joint].sort_values(by="position")
    plt.plot(
        joint_df["position"],
        joint_df["force"],
        marker="o",
        label=joint
    )

plt.xlabel("Position")
plt.ylabel("Force")
plt.title("Force vs Position (All Joints)")
plt.legend()
plt.grid(True)
plt.savefig("all_joints_force_vs_position.png")
plt.show()


# Torque vs Position 
plt.figure(figsize=(10, 6))

for joint in joints_name:
    joint_df = df[df["joint_name"] == joint].sort_values(by="position")
    plt.plot(
        joint_df["position"],
        joint_df["torque"],
        marker="o",
        label=joint
    )

plt.xlabel("Position")
plt.ylabel("Torque")
plt.title("Torque vs Position (All Joints)")
plt.legend()
plt.grid(True)
plt.savefig("all_joints_torque_vs_position.png")
plt.show()