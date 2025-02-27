import os
import pandas as pd
import matplotlib.pyplot as plt

# Folder containing the CSV files
stats_folder = "./"

# Lists to store extracted data
models = []
tokens_per_second = []
energy_per_token = []

# Read each CSV file
for filename in sorted(os.listdir(stats_folder)):
    print(filename)
    if filename.endswith("_stats.csv"):
        model_name = filename.replace("_stats.csv", "")
        file_path = os.path.join(stats_folder, filename)

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Extract relevant values
        tps = float(df.loc[df["Metric"] == "Image Generation Time (s)", "Value"])
        ept = float(df.loc[df["Metric"] == "Total Energy (J)", "Value"])

        # Store the values
        models.append(model_name)
        tokens_per_second.append(tps)
        energy_per_token.append(ept)

# Plot Tokens per Second
plt.figure(figsize=(10, 6))
plt.bar(models, tokens_per_second, color="royalblue")
plt.xlabel("Model")
plt.ylabel("Image Generation Time (s)")
plt.title("Image Generation Time for Each Model")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("image_generation_time.png")
plt.show()

# Plot Energy per Token
plt.figure(figsize=(8, 5))
plt.bar(models, energy_per_token, color="crimson")
plt.xlabel("Model")
plt.ylabel("Total Energy (J)")
plt.title("Total Energy (J) for Each Model")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout() 
plt.savefig("total_energy.png")
plt.show()
