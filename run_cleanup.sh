#!/bin/bash
docker compose down -v

# Remove specific images
docker rmi powerapi/hwpc-sensor powerapi/smartwatts-formula influxdb:2 mongo grafana/grafana
sudo rm -rf /tmp/powerapi-influx/