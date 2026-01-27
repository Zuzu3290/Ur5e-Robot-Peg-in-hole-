import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming your CSV file is saved as contact_sensor_data.csv
df = pd.read_csv("contact_sensor_data.csv", sep="|")

sns.set(style="whitegrid", palette="muted", font_scale=1.2)

plt.figure(figsize=(12, 6))

# Plot each phase separately for clarity
phases = df['phase'].unique()

for phase in phases:
    phase_df = df[df['phase'] == phase]
    plt.plot(
        phase_df['contact_force_N'],
        phase_df['timestamp'],
        marker='o',
        label=phase
    )

plt.xlabel("Time (s)")
plt.ylabel("Contact Force (N)")
plt.title("UR5e Contact Force over Time by Phase")
plt.legend()
plt.tight_layout()
output_path = "contact_force_plot.png"
plt.savefig(output_path, dpi=300)
plt.show()