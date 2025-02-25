from influxdb_client import InfluxDBClient
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter
import numpy as np

client = InfluxDBClient(
   url="http://localhost:8086",
   token="L1y-0yiNX1r5OcFlpaf45R8dOyCsvhMMg4zY9IOC5oKvmFaXgpjKCKaqjdMrEqglEbyKCCRsuEKdHSIobl2hXA==",
   org="org_test"
)

query = '''
    from(bucket:"power_consumption")
    |> range(start: -5m)
    |> filter(fn: (r) => r["_measurement"] == "power_consumption")
    |> filter(fn: (r) => r["target"] == "global")
    |> filter(fn: (r) => r["_field"] == "power")
'''

result = client.query_api().query_data_frame(query)
grouped = result.groupby('_time')['_value'].mean().reset_index()

# Convert UTC to CET (+1)
grouped['_time'] = grouped['_time'] + pd.Timedelta(hours=1)

# Load the CSV power data
csv_power = pd.read_csv('power_measurements.csv')
csv_power['timestamp'] = pd.to_datetime(csv_power['timestamp'])

# Load model timing data with millisecond precision
timing_df = pd.read_csv('inference_timing.csv')
timing_df['start_time'] = pd.to_datetime(timing_df['start_time'], format='%Y-%m-%d %H:%M:%S.%f')
timing_df['end_time'] = pd.to_datetime(timing_df['end_time'], format='%Y-%m-%d %H:%M:%S.%f')

# Create the plot
plt.rcParams['figure.facecolor'] = 'white'
plt.figure(figsize=(12, 6))

# Plot both power consumption datasets
plt.plot(grouped['_time'], grouped['_value'], linewidth=2, color='#2077B4', label='CPU Power')
plt.plot(csv_power['timestamp'], csv_power['pdu_power'], linewidth=2, color='#FF7F0E', label='PDU Power')

plt.xlabel('Time (HH:MM:SS)', fontsize=12)
plt.ylabel('Power (Watts)', fontsize=12)
plt.title('Global Power Consumption', fontsize=14, pad=20)
plt.legend()

plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
plt.grid(True, linestyle='--', alpha=0.7)
plt.margins(x=0.01)

plt.tight_layout()
plt.savefig('power_plot.png', dpi=300, bbox_inches='tight')