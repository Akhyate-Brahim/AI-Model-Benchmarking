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