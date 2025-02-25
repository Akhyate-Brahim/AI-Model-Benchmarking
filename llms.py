import subprocess
import time
from ollama import chat

# Select your model
MODEL_NAME = input("Enter the model name : ")

# Input prompt
PROMPT = "Explain quantum mechanics in simple terms."

# Function to get current GPU power usage
def get_gpu_power():
    """Fetches the current GPU power consumption using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        power_values = [float(x) for x in result.stdout.strip().split("\n")]
        return sum(power_values) / len(power_values)  
    except Exception as e:
        print(f"Error fetching GPU power: {e}")
        return 0

# Start measuring
start_time = time.time()
total_power = 0
samples = 0

# Start text generation
stream = chat(
    model=MODEL_NAME,
    messages=[{'role': 'user', 'content': PROMPT}],
    stream=True,
)

token_count = 0

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
    token_count += 1  
    power = get_gpu_power()
    total_power += power
    samples += 1

# End timing
end_time = time.time()
duration = end_time - start_time

# Compute final stats
average_power = total_power / samples if samples > 0 else 0
total_energy = average_power * duration / 1000  # Convert W * s to kWh
tokens_per_second = token_count / duration if duration > 0 else 0

# Display results
print(f"Total tokens generated: {token_count}")
print(f"Time taken: {duration:.2f} seconds")
print(f"Tokens per second: {tokens_per_second:.2f}")
print(f"Average GPU power: {average_power:.2f} W")
print(f"Total energy consumed: {total_energy:.6f} kWh")


