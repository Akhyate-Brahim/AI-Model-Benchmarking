import subprocess
import time
import csv
from ollama import chat
from power_monitor import power_monitor
import threading

# Models to test
MODEL_NAMES = ["llama3.2", "deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:8b"]
PROMPT = "Explain quantum mechanics in simple terms."

def run_model(MODEL_NAME):
    """Runs a model and returns token count and time taken."""
    start_time = time.time()
    token_count = 0
    stream = chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": PROMPT}],
        stream=True,
    )
    
    for chunk in stream:
        token_count += 1
    
    end_time = time.time()
    duration = end_time - start_time
    tokens_per_second = token_count / duration if duration > 0 else 0
    
    return token_count, duration, tokens_per_second

def log_results(model_name, power_data):
    """Logs power usage and model performance results to separate CSV files."""
    
    power_filename = f"{model_name}_power.csv"
    # Write power data
    with open(power_filename, mode="w", newline="") as power_file:
        writer = csv.writer(power_file)
        writer.writerow(["Time (s)", "GPU Power (W)", "PDU Power (W)"])
        writer.writerows([[f"{t:.1f}", gpu, pdu] for t, gpu, pdu in power_data])

def log_stats(model_name,power_data, token_count, duration, tokens_per_second):
    """Logs power usage, model performance, and energy per token to a CSV file."""
    stats_filename = f"{model_name}_stats.csv"

    total_energy_joules = sum(gpu * 0.1 for _, gpu, pdu in power_data) 
    energy_per_token = total_energy_joules / token_count if token_count > 0 else 0

    # Write model statistics
    with open(stats_filename, mode="w", newline="") as stats_file:
        writer = csv.writer(stats_file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Tokens", token_count])
        writer.writerow(["Duration (s)", f"{duration:.2f}"])
        writer.writerow(["Tokens per second", f"{tokens_per_second:.2f}"])
        writer.writerow(["Total Energy (J)", f"{total_energy_joules:.2f}"])
        writer.writerow(["Energy per Token (J/token)", f"{energy_per_token:.2f}"])

def main():
    """Runs the power measurement, then each model sequentially."""
    for MODEL_NAME in MODEL_NAMES:
        print(f"\nRunning model: {MODEL_NAME}")
        
        # Initialize monitoring variables
        monitoring = True
        power_data = []
        
        # Start power monitoring in a separate thread
        monitor_thread = threading.Thread(
            target=power_monitor, 
            args=(lambda: monitoring, power_data), 
            daemon=True
        )
        monitor_thread.start()
        
        # Wait in idle state (recording these measurements)
        print("Idle period before model run...")
        time.sleep(5)
        
        # Mark the start index of the active model run
        model_start_index = len(power_data)
        
        # Run the model and gather performance stats
        token_count, duration, tokens_per_second = run_model(MODEL_NAME)
        
        # Mark the end index of the active model run
        model_end_index = len(power_data)
        
        # Enter post-run idle state (still recording measurements)
        print(f"Idle period after model run...")
        time.sleep(25)
        
        # Stop monitoring
        monitoring = False
        monitor_thread.join()
        
        # Extract just the model run data for statistics
        model_run_power_data = power_data[model_start_index:model_end_index]
        
        print(f"Total power measurements: {len(power_data)}")
        print(f"Model-only measurements: {len(model_run_power_data)}")
        
        # Log all power data to CSV for visualization
        log_results(MODEL_NAME, power_data)
        
        # Log statistics using only the model run period
        log_stats(MODEL_NAME, model_run_power_data, token_count, duration, tokens_per_second)


if __name__ == "__main__":
    main()

