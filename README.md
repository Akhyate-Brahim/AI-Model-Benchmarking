# AI-Model-Benchmarking

## PowerAPI

#### PowerAPI Setup

- First run the docker compose:
`docker-compose up -d`

- Then get the InfluxDB token via the InfluxDB UI on `http://localhost:8086`

- Update your smartwatts-config.json with the token (replace YOUR_TOKEN)

- Restart the stack with the updated config
```
docker-compose down
docker-compose up -d
```

For full details, see the PowerAPI guide: **https://powerapi.org/reference/overview/**

#### AI Benchmarking
- Script comparing multiple CNN models [cnn_comparison.py](scripts/cnn_comparison.py)

- Script used to plot the power consumption of each model, using PDU power data and PowerAPI data [record_power.py](scripts/record_power.py), make sure to replace the token
## Power Measurement: nvidia-smi & PDU

## Overview  

`llms.py` benchmarks **LLMs (Large Language Models)** by measuring their power consumption while running. It records **GPU power usage** (via `nvidia-smi`) and **PDU power consumption** (via `snmpget`), logs the results, and calculates efficiency metrics like **energy per token**.  

## Features  

- **Power Monitoring**: Continuously records GPU and PDU power usage in a background thread.  
- **Model Execution**: Runs each AI model sequentially while measuring power consumption.  
- **Energy Metrics**: Computes total energy consumption and energy per token.  
- **CSV Logging**: Stores power consumption and model performance data for analysis.  
- **Idle State Measurement**: Includes idle power logs before and after model execution for comparison.  

## How It Works  

1. **Power Monitoring**  
   - Runs in a separate thread, continuously logging power consumption.  
   - Uses `nvidia-smi` to get GPU power draw.  
   - Uses `snmpget` to fetch PDU power usage.  
   - Records timestamps to match power data with execution time.  

2. **Model Execution**  
   - Loops through a list of models (`MODEL_NAMES`).  
   - Starts power monitoring before running a model.  
   - Runs the model inference and tracks token count & duration.  
   - Stops monitoring after model execution.  

3. **Data Logging & Statistics**  
   - Saves power consumption data in a CSV file (`<model_name>_power.csv`).  
   - Computes **energy per token** from power readings.  
   - Logs model statistics separately in `<model_name>_stats.csv`.  



