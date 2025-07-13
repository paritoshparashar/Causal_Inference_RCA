import logging
import numpy as np
from .config import Config  # Assuming Config is defined in config.py
from typing import List, Dict, Tuple

logger = logging.getLogger("anomaly-detector")

class AnomalyDetector:
    def __init__(self, latency_percentile: int):
        self.latency_percentile = latency_percentile

    def _calculate_threshold(self, traces: List[Dict]) -> Tuple[float, List[float]]:
        """
        Calculate the latency threshold and return all latencies
        
        Args:
            traces: List of traces to analyze
            
        Returns:
            Tuple[float, List[float]]: Threshold value and list of all latencies
        """
        latencies = [trace.get("duration", 0) for trace in traces]
        threshold = np.percentile(latencies, self.latency_percentile)
        logger.info(f"Latency threshold ({self.latency_percentile}th percentile): {threshold:.2f}")
        return threshold, latencies

    def detect(self, traces: List[Dict]) -> List[Dict]:
        """
        Detect anomalous traces based on latency threshold
        
        Args:
            traces: List of traces to analyze
            
        Returns:
            List[Dict]: List of anomalous traces
        """
        if not traces:
            return []
            
        # Calculate threshold
        threshold, _ = self._calculate_threshold(traces)

        # Filter traces that exceed the threshold
        return [trace for trace in traces if trace.get("duration", 0) > threshold]
    
    def get_normal_traces(self, traces: List[Dict]) -> List[Dict]:
        """
        Get normal (non-anomalous) traces based on latency threshold
        
        Args:
            traces: List of traces to analyze
            
        Returns:
            List[Dict]: List of normal traces
        """
        if not traces:
            return []
            
        # Calculate threshold
        threshold, _ = self._calculate_threshold(traces)
        
        # Filter traces that are below or at the threshold
        normal_traces = [trace for trace in traces if trace.get("duration", 0) <= threshold]
        
        logger.info(f"Found {len(normal_traces)} normal traces out of {len(traces)} total traces")
        return normal_traces