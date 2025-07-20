"""
Trace Generator

Main interface for generating synthetic OTEL traces with controlled anomalies.
"""

import json
import random
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pathlib import Path

from .service_topology import ServiceTopology
from .latency_calculator import LatencyCalculator


class TraceGenerator:
    """
    Generates synthetic OTEL traces in Jaeger format with controlled latency anomalies.
    
    Features:
    - 10-service microservice topology
    - 90% normal traces, 10% anomalous traces
    - Graduated anomaly severity (1.5x, 1.75x, 2x, 2.5x)
    - Realistic parent-child span relationships
    - OTEL Jaeger format output
    """
    
    def __init__(self):
        self.topology = ServiceTopology()
        self.latency_calculator = LatencyCalculator(self.topology)
        
    def generate_traces(self, num_traces: int, slow_service: str, 
                       output_file: str = None) -> str:
        """
        Generate synthetic traces with controlled anomalies.
        
        Args:
            num_traces: Total number of traces to generate
            slow_service: Which service to slow down in anomalous traces
            output_file: Output file path (optional)
            
        Returns:
            Path to generated trace file
        """
        if not self.topology.validate_service_name(slow_service):
            raise ValueError(f"Invalid service name: {slow_service}. "
                           f"Available services: {self.topology.get_service_names()}")
        
        if output_file is None:
            output_file = f"output/traces/synthetic_traces_{slow_service}_{num_traces}.json"
        
        print(f"Generating {num_traces} synthetic traces...")
        print(f"Target slow service: {slow_service}")
        print(f"Distribution: 90% normal, 10% anomalous (1.5x-2.5x slower)")
        
        # Generate all traces
        traces = []
        stats = {'normal': 0, 'anomalous': 0}
        
        for i in range(num_traces):
            # Determine if this trace should be normal or anomalous
            trace_type = self.latency_calculator.determine_trace_type(i, num_traces)
            
            # Generate the trace
            trace = self._generate_single_trace(slow_service, trace_type)
            traces.append(trace)
            
            stats[trace_type] += 1
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_traces} traces...")
        
        # Save traces to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(traces, f, indent=2)
        
        print(f"\\n✓ Generated {num_traces} traces:")
        print(f"  Normal traces: {stats['normal']}")
        print(f"  Anomalous traces: {stats['anomalous']}")
        print(f"  Saved to: {output_path}")
        
        return str(output_path)
    
    def _generate_single_trace(self, slow_service: str, trace_type: str) -> Dict:
        """Generate a single trace with all spans."""
        trace_id = self._generate_trace_id()
        
        # Calculate latencies for all services
        service_latencies = self.latency_calculator.calculate_trace_latencies(
            slow_service if trace_type == 'anomalous' else None,
            trace_type
        )
        
        # Generate spans in call order
        spans = self._generate_spans(trace_id, service_latencies)
        
        # Build processes map
        processes = self._build_processes()
        
        return {
            "traceID": trace_id,
            "spans": spans,
            "processes": processes
        }
    
    def _generate_spans(self, trace_id: str, service_latencies: Dict[str, float]) -> List[Dict]:
        """Generate spans for all services with proper parent-child relationships."""
        spans = []
        span_id_map = {}  # service_name -> span_id
        
        # Get services in call order (depth-first traversal)
        call_chain = self.topology.get_call_chain('frontend')
        
        # Generate base timestamp
        base_time = datetime.now(timezone.utc)
        current_time_us = int(base_time.timestamp() * 1_000_000)
        
        for service_name in call_chain:
            config = self.topology.get_service_config(service_name)
            span_id = self._generate_span_id()
            span_id_map[service_name] = span_id
            
            # Calculate span timing
            duration_ms = service_latencies[service_name]
            duration_us = int(duration_ms * 1000)  # Convert to microseconds
            
            # Determine parent span
            parent_span_id = None
            references = []
            
            # Find direct upstream service (parent)
            for upstream_service in self.topology.get_service_names():
                if service_name in self.topology.get_downstream_services(upstream_service):
                    parent_span_id = span_id_map.get(upstream_service)
                    if parent_span_id:
                        references.append({
                            "refType": "CHILD_OF",
                            "traceID": trace_id,
                            "spanID": parent_span_id
                        })
                    break
            
            # Create span
            span = {
                "traceID": trace_id,
                "spanID": span_id,
                "flags": 1,
                "operationName": config.operation_name,
                "references": references,
                "startTime": current_time_us,
                "duration": duration_us,
                "tags": [
                    {
                        "key": "service.name",
                        "type": "string", 
                        "value": service_name
                    },
                    {
                        "key": "span.kind",
                        "type": "string",
                        "value": "server"
                    }
                ],
                "processID": f"p{len(spans) + 1}",
                "warnings": None
            }
            
            spans.append(span)
            
            # Advance time for next span
            current_time_us += duration_us
        
        return spans
    
    def _build_processes(self) -> Dict[str, Dict]:
        """Build the processes map for all services."""
        processes = {}
        
        for i, service_name in enumerate(self.topology.get_service_names()):
            process_id = f"p{i + 1}"
            processes[process_id] = {
                "serviceName": service_name,
                "tags": [
                    {
                        "key": "telemetry.sdk.language",
                        "type": "string",
                        "value": "python"
                    },
                    {
                        "key": "telemetry.sdk.name",
                        "type": "string",
                        "value": "opentelemetry"
                    }
                ]
            }
        
        return processes
    
    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return uuid.uuid4().hex
    
    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return uuid.uuid4().hex[:16]  # Shorter span IDs
    
    def get_topology_info(self) -> str:
        """Get information about the service topology."""
        return self.topology.get_topology_summary()
    
    def get_available_services(self) -> List[str]:
        """Get list of all available services that can be slowed down."""
        return self.topology.get_service_names()


def generate_traces(num_traces: int, slow_service: str, output_file: str = None) -> str:
    """
    Convenience function to generate synthetic traces.
    
    Args:
        num_traces: Total number of traces to generate
        slow_service: Which service to slow down in anomalous traces
        output_file: Output file path (optional)
        
    Returns:
        Path to generated trace file
    """
    generator = TraceGenerator()
    return generator.generate_traces(num_traces, slow_service, output_file)
