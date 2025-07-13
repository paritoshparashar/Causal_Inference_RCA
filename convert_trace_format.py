#!/usr/bin/env python3
"""
Convert TestingTraces.json format to Jaeger format for dependency analysis.

TestingTraces format uses:
- trace_id, span_id, parent_id, service, start_time, duration_ms

Jaeger format expects:
- traceID, spanID, references, operationName, duration (microseconds), startTime
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def convert_testing_traces_to_jaeger(input_file: str, output_file: str = None) -> str:
    """
    Convert TestingTraces.json format to Jaeger format.
    
    Args:
        input_file: Path to TestingTraces.json
        output_file: Output path (optional, defaults to input_file with _jaeger suffix)
        
    Returns:
        Path to converted file
    """
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_jaeger{input_path.suffix}")
    
    print(f"Converting {input_file} to Jaeger format...")
    
    # Load testing traces
    with open(input_file, 'r') as f:
        testing_traces = json.load(f)
    
    jaeger_traces = []
    
    for trace in testing_traces:
        trace_id = trace['trace_id']
        spans = trace['spans']
        
        # Build service to process ID mapping
        unique_services = list(set(span['service'] for span in spans))
        service_to_process_id = {service: f"p{i+1}" for i, service in enumerate(unique_services)}
        
        # Convert spans to Jaeger format
        jaeger_spans = []
        
        for span in spans:
            # Parse start time to microseconds
            start_time_str = span['start_time']
            start_dt = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            start_time_us = int(start_dt.timestamp() * 1_000_000)
            
            # Convert duration from ms to microseconds
            duration_us = span['duration_ms'] * 1000
            
            # Build references for parent-child relationships
            references = []
            if span['parent_id']:
                references.append({
                    "refType": "CHILD_OF",
                    "traceID": trace_id,
                    "spanID": span['parent_id']
                })
            
            # Get process ID for this service
            process_id = service_to_process_id[span['service']]
            
            # Create Jaeger span
            jaeger_span = {
                "traceID": trace_id,
                "spanID": span['span_id'],
                "flags": 1,
                "operationName": f"{span['service']}Service_Operation",  # Create operation name from service
                "references": references,
                "startTime": start_time_us,
                "duration": duration_us,
                "tags": [
                    {
                        "key": "service.name",
                        "type": "string",
                        "value": span['service']
                    }
                ],
                "processID": process_id,
                "warnings": None
            }
            
            jaeger_spans.append(jaeger_span)
        
        # Build processes map
        processes = {}
        for service, process_id in service_to_process_id.items():
            processes[process_id] = {
                "serviceName": service,
                "tags": [
                    {
                        "key": "telemetry.sdk.language",
                        "type": "string",
                        "value": "python"
                    }
                ]
            }
        
        # Create Jaeger trace
        jaeger_trace = {
            "traceID": trace_id,
            "spans": jaeger_spans,
            "processes": processes
        }
        
        jaeger_traces.append(jaeger_trace)
    
    # Save converted traces
    with open(output_file, 'w') as f:
        json.dump(jaeger_traces, f, indent=2)
    
    print(f"✓ Converted {len(jaeger_traces)} traces with {sum(len(t['spans']) for t in jaeger_traces)} spans")
    print(f"✓ Saved to: {output_file}")
    
    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_trace_format.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result_file = convert_testing_traces_to_jaeger(input_file, output_file)
        print(f"\n🎉 Conversion completed successfully!")
        print(f"You can now run: python main.py --analyze-only {result_file}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
