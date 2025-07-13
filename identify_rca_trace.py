#!/usr/bin/env python3
"""
Identify the specific trace used in RCA analysis
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

def extract_service_name(operation_name: str) -> str:
    """Extract service name from operation name."""
    if 'Server_' in operation_name:
        return operation_name.split('Server_')[0].replace('Service', '').lower()
    elif 'Client_' in operation_name:
        return operation_name.split('Client_')[0].replace('Service', '').lower()
    else:
        return operation_name.lower().replace('service', '')

def analyze_traces():
    # Load trace data
    with open('output/traces/collected_traces_20250713_230056.json', 'r') as f:
        trace_data = json.load(f)
    
    print(f"Loaded {len(trace_data)} traces")
    
    # Extract latencies per trace
    latency_data = {}
    trace_ids = []
    
    for trace in trace_data:
        trace_id = trace['traceID']
        trace_ids.append(trace_id)
        
        # Extract latencies for each span/service in this trace
        trace_latencies = {}
        
        for span in trace['spans']:
            service_name = extract_service_name(span['operationName'])
            duration_us = span.get('duration', 0)
            duration_ms = duration_us / 1000.0
            
            if service_name in trace_latencies:
                trace_latencies[service_name] += duration_ms
            else:
                trace_latencies[service_name] = duration_ms
        
        # Store this trace's latencies
        for service, latency in trace_latencies.items():
            if service not in latency_data:
                latency_data[service] = []
            latency_data[service].append(latency)
    
    # Convert to DataFrame
    max_length = max(len(latencies) for latencies in latency_data.values())
    
    for service in latency_data:
        while len(latency_data[service]) < max_length:
            latency_data[service].append(np.nan)
    
    df = pd.DataFrame(latency_data)
    df = df.dropna()
    
    print(f"After filtering complete traces: {len(df)} traces")
    print(f"Services: {list(df.columns)}")
    
    # Calculate total latencies and find anomalous traces
    total_latencies = df.sum(axis=1)
    threshold = np.percentile(total_latencies, 95.0)
    
    anomalous_mask = total_latencies > threshold
    anomalous_data = df[anomalous_mask]
    
    print(f"\nThreshold (95th percentile): {threshold:.2f}ms")
    print(f"Anomalous traces: {len(anomalous_data)}")
    
    # Show the first anomalous trace (used in RCA)
    if len(anomalous_data) > 0:
        first_anomalous = anomalous_data.head(1)
        first_idx = first_anomalous.index[0]
        
        print(f"\n=== FIRST ANOMALOUS TRACE (used in RCA) ===")
        print(f"DataFrame Index: {first_idx}")
        print(f"Trace ID (estimated): {trace_ids[first_idx]}")
        print(f"Service latencies:")
        
        for service, latency in first_anomalous.iloc[0].items():
            print(f"  {service}: {latency:.2f}ms")
        
        total = first_anomalous.sum(axis=1).iloc[0]
        frontend_latency = first_anomalous.iloc[0].get('frontend', 0)
        
        print(f"Total latency: {total:.2f}ms")
        print(f"Frontend latency: {frontend_latency:.2f}ms")
        
        # This should match the RCA results
        print(f"\nRCA Expected Values:")
        print(f"  Frontend outlier value: 24670.706ms")
        print(f"  Frontend actual value:  {frontend_latency:.3f}ms")
        print(f"  Match: {'✓' if abs(frontend_latency - 24670.706) < 1 else '✗'}")
        
        print(f"\n=== ALL ANOMALOUS TRACES ===")
        for i, (idx, row) in enumerate(anomalous_data.iterrows()):
            total = row.sum()
            frontend_lat = row.get('frontend', 0)
            trace_id = trace_ids[idx] if idx < len(trace_ids) else "Unknown"
            print(f"Trace {i+1} (index {idx}, ID: {trace_id[:16]}...): Total={total:.2f}ms, Frontend={frontend_lat:.2f}ms")

if __name__ == "__main__":
    analyze_traces()
