import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the data
def load_joint_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="|")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# Extract phase intervals from the DataFrame
def extract_phase_intervals(df: pd.DataFrame):
    phase_intervals = []
    current_phase = df.iloc[0]["phase"]
    start_time = df.iloc[0]["timestamp"]

    for i in range(1, len(df)):
        if df.iloc[i]["phase"] != current_phase:
            end_time = df.iloc[i - 1]["timestamp"]
            phase_intervals.append((current_phase, start_time, end_time))
            current_phase = df.iloc[i]["phase"]
            start_time = df.iloc[i]["timestamp"]

    phase_intervals.append((current_phase, start_time, df.iloc[-1]["timestamp"]))
    return phase_intervals

# General plotting function
def plot_joint_data(df: pd.DataFrame, joints_name: list, value_column: str, 
                    ylabel: str, line_color: str, phase_colors: dict):
    
    phase_intervals = extract_phase_intervals(df)
    
    fig, axes = plt.subplots(nrows=len(joints_name), ncols=1, figsize=(16, 12), sharex=True)

    for ax, joint in zip(axes, joints_name):
        joint_df = df[df["joint_name"] == joint]

        # Phase shading
        for phase, t_start, t_end in phase_intervals:
            ax.axvspan(t_start, t_end, color=phase_colors.get(phase, "#EEEEEE"), alpha=0.5, zorder=0)

        # Plot joint data
        ax.plot(joint_df["timestamp"], joint_df[value_column], color=line_color, linewidth=1.8, zorder=1)

        ax.set_ylabel(ylabel)
        ax.set_title(joint, loc="left", fontsize=10)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    axes[-1].set_xlabel("Time (s)")

    # Phase legend
    legend_patches = [plt.matplotlib.patches.Patch(color=phase_colors[p], label=p) for p in phase_colors]
    fig.legend(handles=legend_patches, loc="upper right", title="Phase")

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.show()

# Main execution
file_path = "joint_data.csv"
df = load_joint_data(file_path)

joints_name = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

phase_colors = {
    "INITIAL": "#D6EAF8",  # light blue
    "ACT_1": "#F5B7B1",     # pinkish red
    "ACT_2": "#ABEBC6",     # green
    "ACT_3": "#F9E79F",     # yellow
}

# Plot Force
plot_joint_data(df, joints_name, value_column="force_N", ylabel="Force (N)", line_color="black", phase_colors=phase_colors)

# Plot Torque
plot_joint_data(df, joints_name, value_column="torque_Nm", ylabel="Torque (Nm)", line_color="purple", phase_colors=phase_colors)
