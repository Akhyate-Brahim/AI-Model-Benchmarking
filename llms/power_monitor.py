import threading
import time
import subprocess
import csv


def get_gpu_metrics():
    """Fetches GPU power consumption and temperature using nvidia-smi, and PDU power."""
    try:
        # Get GPU power and temperature
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Extract float values for power and temperature
        gpu_metrics = [list(map(float, line.split(","))) for line in gpu_result.stdout.strip().split("\n") if line.strip()]
        avg_gpu_power = sum(m[0] for m in gpu_metrics) / len(gpu_metrics) if gpu_metrics else 0
        avg_gpu_temp = sum(m[1] for m in gpu_metrics) / len(gpu_metrics) if gpu_metrics else 0

        # Get PDU power usage
        pdu_result = subprocess.run(
            ["snmpget", "-v2c", "-c", "public", "192.168.10.168", "PowerNet-MIB::ePDUPhaseStatusActivePower.1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Extract numerical power value from SNMP output
        try:
            pdu_power_value = float(pdu_result.stdout.split(":")[-1].strip())
        except ValueError:
            pdu_power_value = 0  # Default to 0 if parsing fails

        return avg_gpu_power, avg_gpu_temp, pdu_power_value

    except Exception as e:
        print(f"Error fetching power/temperature data: {e}")
        return 0, 0, 0  # Default values in case of failure


def power_monitor(is_monitoring,power_data):
    """Runs in a separate thread to continuously record power data."""
    start_time = time.time()
    
    while is_monitoring():
        gpu_power, pdu_power = get_gpu_metrics()
        timestamp = time.time() - start_time
        power_data.append((timestamp, gpu_power, pdu_power))
        time.sleep(0.1)  # Adjust sampling rate if needed

