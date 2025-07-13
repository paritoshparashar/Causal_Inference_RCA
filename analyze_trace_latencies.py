#!/usr/bin/env python3
"""
Extract and print latency information for each trace in TestingTraces_jaeger_fixed.json
"""

import json
import sys
from datetime import datetime


def calculate_trace_duration(trace):
    """
    Calculate the total duration of a trace using multiple methods.
    """
    spans = trace.get("spans", [])
    if not spans:
        return 0.0
    
    # Method 1: Find the root span (span with no parent references)
    root_spans = []
    for span in spans:
        references = span.get("references", [])
        # A root span has no CHILD_OF references
        if not any(ref.get("refType") == "CHILD_OF" for ref in references):
            root_spans.append(span)
    
    if root_spans:
        # Use the longest root span duration
        return max(span.get("duration", 0) for span in root_spans)
    
    # Method 2: Calculate span of entire trace (earliest start to latest end)
    if spans:
        start_times = [span.get("startTime", 0) for span in spans]
        end_times = [span.get("startTime", 0) + span.get("duration", 0) for span in spans]
        
        if start_times and end_times:
            total_duration = max(end_times) - min(start_times)
            return total_duration
    
    # Method 3: Fallback - sum all span durations (may double-count)
    return sum(span.get("duration", 0) for span in spans)


def analyze_trace_latencies(file_path):
    """
    Analyze and print latency information for all traces.
    """
    print(f"📊 Analyzing trace latencies from: {file_path}")
    print("=" * 80)
    
    # Load traces
    with open(file_path, 'r') as f:
        traces = json.load(f)
    
    print(f"Total traces found: {len(traces)}")
    print("=" * 80)
    
    total_latency = 0
    latencies = []
    
    for i, trace in enumerate(traces, 1):
        trace_id = trace.get("traceID", "unknown")
        trace_duration_us = calculate_trace_duration(trace)
        trace_duration_ms = trace_duration_us / 1000.0  # Convert to milliseconds
        
        latencies.append(trace_duration_ms)
        total_latency += trace_duration_ms
        
        # Get service information from spans
        services = set()
        span_count = len(trace.get("spans", []))
        
        for span in trace.get("spans", []):
            # Extract service name from processID
            process_id = span.get("processID", "")
            processes = trace.get("processes", {})
            if process_id in processes:
                service_name = processes[process_id].get("serviceName", "unknown")
                services.add(service_name)
        
        services_str = ", ".join(sorted(services)) if services else "none"
        
        print(f"Request {i:2d}: {trace_duration_ms:8.2f} ms | "
              f"Spans: {span_count:2d} | "
              f"Services: {services_str}")
        print(f"           TraceID: {trace_id}")
        print()
    
    # Calculate statistics
    avg_latency = total_latency / len(traces) if traces else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    
    # Sort latencies for percentiles
    sorted_latencies = sorted(latencies)
    
    def percentile(data, p):
        if not data:
            return 0
        index = (len(data) - 1) * p / 100
        if index == int(index):
            return data[int(index)]
        else:
            lower = data[int(index)]
            upper = data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    p50 = percentile(sorted_latencies, 50)
    p95 = percentile(sorted_latencies, 95)
    p99 = percentile(sorted_latencies, 99)
    
    print("=" * 80)
    print("📈 LATENCY STATISTICS")
    print("=" * 80)
    print(f"Total requests:     {len(traces)}")
    print(f"Average latency:    {avg_latency:8.2f} ms")
    print(f"Minimum latency:    {min_latency:8.2f} ms")
    print(f"Maximum latency:    {max_latency:8.2f} ms")
    print(f"50th percentile:    {p50:8.2f} ms")
    print(f"95th percentile:    {p95:8.2f} ms")
    print(f"99th percentile:    {p99:8.2f} ms")
    print()
    
    # Identify potential anomalies (requests significantly higher than average)
    threshold = avg_latency * 2  # Simple threshold: 2x average
    anomalies = [(i+1, lat) for i, lat in enumerate(latencies) if lat > threshold]
    
    if anomalies:
        print("🚨 POTENTIAL ANOMALIES (>2x average latency):")
        print("-" * 50)
        for req_num, latency in anomalies:
            print(f"Request {req_num:2d}: {latency:8.2f} ms ({latency/avg_latency:.1f}x average)")
    else:
        print("✅ No significant anomalies detected (all requests within 2x average)")
    
    print("=" * 80)


if __name__ == "__main__":
    file_path = "output/traces/TestingTraces_jaeger_fixed.json"
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    try:
        analyze_trace_latencies(file_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
