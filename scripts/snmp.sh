#!/bin/bash

PDU_IP="192.168.10.168"  # IP address of the PDU
COMMUNITY="public"        # SNMP community string
OID="PowerNet-MIB::ePDUPhaseStatusActivePower.1"  # OID for the power consumption
OUTPUT_FILE="${1:-power_measurements.csv}"  # Default file if no argument is provided
INTERVAL=1  # Interval in seconds between queries

# Create a header in the output file if it's empty
if [ ! -f "$OUTPUT_FILE" ]; then
  echo "timestamp,pdu_power" > "$OUTPUT_FILE"
fi

while true; do
  # Get current timestamp
  timestamp=$(date "+%Y-%m-%d %H:%M:%S")

  # Get PDU power measurement
  pdu_power=$(snmpget -v2c -c "$COMMUNITY" "$PDU_IP" "$OID" | awk -F" " '{print $4}')

  # Append the data to the file
  echo "$timestamp,$pdu_power" >> "$OUTPUT_FILE"
  
  # Print to console
  echo "$timestamp | PDU: $pdu_power W"

  # Wait for the next interval
  sleep "$INTERVAL"
done
