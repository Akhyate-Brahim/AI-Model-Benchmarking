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
We use `unified_measure.sh` to measure power consumption:
- GPU Power: Retrieved using nvidia-smi.
- System Power: Measured via PDU (SNMP protocol) using powernet45.mib.
- Storage: Power data is logged into a CSV file.
### Comparing LLMs:
- Script: llms.py
- Purpose: Benchmarks different LLMs by answering the same question and measuring:
    - Generation speed
    - Power consumption during inference
Power Measurement: Uses unified_measure.sh to track both GPU and system power usage.
### Comparing Image Generation Models
- Script: image_generation.py
- Purpose: Generates images using different models with the same prompt to compare performance and power efficiency.
