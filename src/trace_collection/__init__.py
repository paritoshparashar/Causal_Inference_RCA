"""
Trace Collection Package

This package provides functionality for collecting traces from Jaeger
and detecting anomalies based on latency patterns.
"""

from .config import Config
from .jaeger_client import JaegerClient
from .anomaly_detection import AnomalyDetector

__all__ = ['Config', 'JaegerClient', 'AnomalyDetector']
