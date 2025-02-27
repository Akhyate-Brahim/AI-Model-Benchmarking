import time
import csv
import threading
import torch
from diffusers import StableDiffusionPipeline
from power_monitor import power_monitor  # Assumed external function

# Models to test
MODEL_NAMES = ["stabilityai/stable-diffusion-2-1", "runwayml/stable-diffusion-v1-5"]
PROMPT = "A highly detailed portrait of a female astronaut in a futuristic space station, wearing a sleek, high-tech spacesuit, looking out into the galaxy."

# Experiment parameters
NUM_RUNS = 3  # Number of times to repeat the experiment per model

def run_model(MODEL_NAME, batch_size):
    """Runs a Stable Diffusion model with a given batch size and returns time per image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model in half precision (FP16) for efficiency
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME).to(device)
    pipe.to(torch.float16)  

    torch.manual_seed(42)  # Fixed seed for reproducibility

    start_time = time.time()
    images = pipe(PROMPT, num_images_per_prompt=batch_size).images
    end_time = time.time()

    # Save images for quality comparison
    safe_model_name = MODEL_NAME.split("/")[-1]
    for i, img in enumerate(images):
        img.save(f"{safe_model_name}_batch{batch_size}_img{i+1}.png")

    return (end_time - start_time) / batch_size  # Normalize by batch size


def log_results(model_name, batch_size, power_data):
    """Logs power data to a CSV file with batch size."""
    safe_model_name = model_name.split("/")[-1]
    filename = f"{safe_model_name}_batch{batch_size}_power.csv"

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "GPU Power (W)", "GPU Temp (°C)", "PDU Power (W)"])
        writer.writerows([[f"{t:.1f}", gpu, temp, pdu] for t, gpu, temp, pdu in power_data])

def log_stats(model_name, batch_size, power_data, durations):
    """Logs energy consumption per image with batch size."""
    safe_model_name = model_name.split("/")[-1]
    filename = f"images/{safe_model_name}_batch{batch_size}_stats.csv"
    
    avg_duration = sum(durations) / len(durations)
    
    # Compute energy in joules: P(W) * time(s)
    total_energy_joules = sum(gpu * 0.1 for _, gpu, _, _ in power_data) / len(durations)
    
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Avg Image Generation Time (s)", f"{avg_duration:.2f}"])
        writer.writerow(["Avg Energy per Image (J)", f"{total_energy_joules:.2f}"])

BATCH_SIZES = [1, 2, 4]

def main():
    """Runs multiple power measurements per model and batch size."""
    for MODEL_NAME in MODEL_NAMES:
        for batch_size in BATCH_SIZES:
            print(f"\nRunning model: {MODEL_NAME} with batch size {batch_size}")

            durations = []
            power_data = []

            for run in range(NUM_RUNS):
                print(f"Trial {run + 1}/{NUM_RUNS}")

                monitoring = True

                monitor_thread = threading.Thread(
                    target=power_monitor, 
                    args=(lambda: monitoring, power_data), 
                    daemon=True
                )
                monitor_thread.start()

                print("Idle period before model run...")
                time.sleep(5)

                duration = run_model(MODEL_NAME, batch_size)

                print("Idle period after model run...")
                time.sleep(25)

                monitoring = False
                monitor_thread.join()

                durations.append(duration)

            log_results(MODEL_NAME, batch_size, power_data)
            log_stats(MODEL_NAME, batch_size, power_data, durations)


if __name__ == "__main__":
    main()
