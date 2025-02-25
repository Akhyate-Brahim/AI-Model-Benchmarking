import matplotlib.pyplot as plt
import pandas as pd
import re
import os

def extract_numeric_power(power_str):
    match = re.search(r'(\d+\.?\d*)', str(power_str), re.IGNORECASE)
    return float(match.group(1)) if match else None

def plot_power_consumption(files, labels, field, title, output_file, steps=40, offsets=None):
    plt.figure(figsize=(10, 5))
    
    for i, file in enumerate(files):
        df = pd.read_csv(file, parse_dates=['timestamp'])
        times = df['timestamp']
        seconds = (times - times.iloc[0]).dt.total_seconds()
        
        if field in df.columns:
            offset = offsets[i] if offsets else 0
            plt.plot(seconds[:steps], df[field][offset:steps+offset], label=labels[i])
        else:
            print(f"Warning: Column '{field}' not found in {file}")
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Power consumption (W)')
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# Get user input
num_files = int(input("Enter the number of files: "))
files = []
for i in range(num_files):
    while True:
        file_name = input(f"Enter file {i+1} name: ")
        if os.path.exists(file_name):
            files.append(file_name)
            break
        else:
            print("File not found. Please enter a valid file name.")

labels = [input(f"Enter label for {files[i]}: ") for i in range(num_files)]
field = input("Enter the field to plot: ")
title = input("Enter the graph title: ")
output_file = input("Enter the output image file name: ")
offsets = [int(input(f"Enter offset for {files[i]} (default 0): ") or 0) for i in range(num_files)]

# Execute the function
plot_power_consumption(files, labels, field, title, output_file, offsets=offsets)