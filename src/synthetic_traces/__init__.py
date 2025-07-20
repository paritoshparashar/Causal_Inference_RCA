"""
Synthetic Trace Generation Module

This module provides functionality to generate synthetic OTEL traces 
with controlled latency anomalies for causal inference testing.

Components:
- ServiceTopology: Defines the 10-service microservice architecture
- LatencyCalculator: Handles latency propagation and anomaly injection
- TraceGenerator: Main interface for generating synthetic traces
"""

from .service_topology import ServiceTopology
from .latency_calculator import LatencyCalculator
from .trace_generator import TraceGenerator, generate_traces

__all__ = ['ServiceTopology', 'LatencyCalculator', 'TraceGenerator', 'generate_traces']
