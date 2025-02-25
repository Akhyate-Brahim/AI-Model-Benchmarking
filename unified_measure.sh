#!/bin/bash

PDU_IP="192.168.10.168"  # IP address of the PDU
COMMUNITY="public"        # SNMP community string
OID="PowerNet-MIB::ePDUPhaseStatusActivePower.1"  # OID for the power consumption
OUTPUT_FILE="${1:-power_measurements.csv}"  # Default file if no argument is provided
INTERVAL=1  # Interval in seconds between queries

# Create a header in the output file if it's empty
if [ ! -f "$OUTPUT_FILE" ]; then
  echo "timestamp,pdu_power,gpu_power,gpu_utilization,gpu_memory" > "$OUTPUT_FILE"
fi

while true; do
  # Get current timestamp
  timestamp=$(date "+%Y-%m-%d %H:%M:%S")

  # Get PDU power measurement
  pdu_power=$(snmpget -v2c -c "$COMMUNITY" "$PDU_IP" "$OID" | awk -F" " '{print $4}')

  # Get GPU power and utilization
  gpu_data=$(nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits)

  # Extract GPU values
  gpu_power=$(echo "$gpu_data" | awk -F"," '{print $1}')
  gpu_utilization=$(echo "$gpu_data" | awk -F"," '{print $2}')
  gpu_memory=$(echo "$gpu_data" | awk -F"," '{print $3}')

  # Append the data to the file
  echo "$timestamp,$pdu_power,$gpu_power,$gpu_utilization,$gpu_memory" >> "$OUTPUT_FILE"
  
  # Print to console
  echo "$timestamp | PDU: $pdu_power W | GPU: $gpu_power W, $gpu_utilization %, $gpu_memory MB"

  # Wait for the next interval
  sleep "$INTERVAL"
done

