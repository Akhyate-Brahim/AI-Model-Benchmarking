from influxdb_client import InfluxDBClient
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter
import numpy as np
import warnings
from influxdb_client.client.warnings import MissingPivotFunction

# Suppress InfluxDB warning
warnings.simplefilter("ignore", MissingPivotFunction)

client = InfluxDBClient(
   url="http://localhost:8086",
   token="YOUR_TOKEN",
   org="org_test"
)

query = '''
    from(bucket:"power_consumption")
    |> range(start: -15m)
    |> filter(fn: (r) => r["_measurement"] == "power_consumption")
    |> filter(fn: (r) => r["target"] == "global")
    |> filter(fn: (r) => r["_field"] == "power")
'''

result = client.query_api().query_data_frame(query)
grouped = result.groupby('_time')['_value'].mean().reset_index()

# Convert UTC to CET (+1)
grouped['_time'] = grouped['_time'] + pd.Timedelta(hours=1)

# IMPORTANT: Convert to naive datetime to match CSV timestamps
grouped['_time'] = grouped['_time'].dt.tz_localize(None)

# Load CSV power data
csv_power = pd.read_csv('power_measurements.csv')
csv_power['timestamp'] = pd.to_datetime(csv_power['timestamp'])

# Get the min and max timestamps from InfluxDB data
min_time = grouped['_time'].min()
max_time = grouped['_time'].max()

# Filter CSV data to the same time range
filtered_csv_power = csv_power[(csv_power['timestamp'] >= min_time) & 
                              (csv_power['timestamp'] <= max_time)]

# Load model timing data with millisecond precision
timing_df = pd.read_csv('inference_timing.csv')
timing_df['start_time'] = pd.to_datetime(timing_df['start_time'], format='%Y-%m-%d %H:%M:%S.%f')
timing_df['end_time'] = pd.to_datetime(timing_df['end_time'], format='%Y-%m-%d %H:%M:%S.%f')

# Create the plot
plt.rcParams['figure.facecolor'] = 'white'
plt.figure(figsize=(12, 6))

# Plot power consumption - both InfluxDB and filtered CSV data
plt.plot(grouped['_time'], grouped['_value'], linewidth=2, color='#2077B4', label='CPU Power')
plt.plot(filtered_csv_power['timestamp'], filtered_csv_power['pdu_power'], 
         linewidth=2, color='#FF7F0E', label='PDU Power')

# Add vertical lines for model inferences
colors = plt.cm.tab10(np.linspace(0, 1, len(timing_df)))
for idx, row in timing_df.iterrows():
    plt.axvline(x=row['start_time'], color=colors[idx], linestyle='--', alpha=0.7)
    plt.axvline(x=row['end_time'], color=colors[idx], linestyle='--', alpha=0.7, label=row['model'])

plt.xlabel('Time (HH:MM:SS)', fontsize=12)
plt.ylabel('Power (Watts)', fontsize=12)
plt.title('Global Power Consumption', fontsize=14, pad=20)

plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
plt.grid(True, linestyle='--', alpha=0.7)
plt.margins(x=0.01)

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Customize appearance
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('power_plot.png', dpi=300, bbox_inches='tight')

# Define a margin of 1 second after each inference period
margin = pd.Timedelta(seconds=2)

# Compute average power during each inference interval with the margin
avg_power = []
for idx, row in timing_df.iterrows():
    mask = (grouped['_time'] >= row['start_time']) & (grouped['_time'] <= (row['end_time'] + margin))
    avg = grouped.loc[mask, '_value'].mean()
    avg_power.append(avg)
timing_df['avg_power'] = avg_power

# Compute average PDU power too
avg_pdu_power = []
for idx, row in timing_df.iterrows():
    mask = (filtered_csv_power['timestamp'] >= row['start_time']) & (filtered_csv_power['timestamp'] <= (row['end_time'] + margin))
    if sum(mask) > 0:  # Only if we have matching data
        avg = filtered_csv_power.loc[mask, 'pdu_power'].mean()
    else:
        avg = np.nan
    avg_pdu_power.append(avg)
timing_df['avg_pdu_power'] = avg_pdu_power

# Scatter plot: x-axis = GFLOPs, y-axis = average power (Watts)
plt.figure(figsize=(8,6))
plt.scatter(timing_df['flops'], timing_df['avg_power'], s=100, color='green', label='CPU Power')

# Add PDU power points if available
if 'avg_pdu_power' in timing_df.columns and not timing_df['avg_pdu_power'].isna().all():
    # Only plot non-NaN PDU power values
    mask = ~timing_df['avg_pdu_power'].isna()
    plt.scatter(timing_df.loc[mask, 'flops'], timing_df.loc[mask, 'avg_pdu_power'], 
                s=100, color='orange', label='PDU Power')

for idx, row in timing_df.iterrows():
    plt.annotate(row['model'], (row['flops'], row['avg_power']),
                 textcoords="offset points", xytext=(5,5), ha='center')



plt.xlabel('GFLOPs')
plt.ylabel('Average Power Consumption (Watts)')
plt.title('Inference Power Consumption vs GFLOPs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('power_vs_gflops.png', dpi=300, bbox_inches='tight')

# Calculate inference duration in seconds
timing_df['duration_seconds'] = (timing_df['end_time'] - timing_df['start_time']).dt.total_seconds()

# Calculate total energy consumed (in joules = watts * seconds)
timing_df['cpu_energy_joules'] = timing_df['avg_power'] * timing_df['duration_seconds']
timing_df['pdu_energy_joules'] = timing_df['avg_pdu_power'] * timing_df['duration_seconds']

# Calculate energy per second
timing_df['cpu_energy_per_second'] = timing_df['cpu_energy_joules'] / timing_df['duration_seconds']
timing_df['pdu_energy_per_second'] = timing_df['pdu_energy_joules'] / timing_df['duration_seconds']

# Select and rename columns for the output CSV
energy_df = timing_df[['model', 'parameters', 'duration_seconds', 
                      'cpu_energy_joules', 'pdu_energy_joules',
                      'cpu_energy_per_second', 'pdu_energy_per_second']]

# Rename columns to be more descriptive
energy_df = energy_df.rename(columns={
    'parameters': 'parameter_count',
    'cpu_energy_joules': 'cpu_total_energy_joules',
    'pdu_energy_joules': 'pdu_total_energy_joules',
    'cpu_energy_per_second': 'cpu_power_watts',
    'pdu_energy_per_second': 'pdu_power_watts'
})

# Save to CSV
energy_df.to_csv('model_energy_consumption.csv', index=False)
print("Energy consumption data saved to model_energy_consumption.csv")