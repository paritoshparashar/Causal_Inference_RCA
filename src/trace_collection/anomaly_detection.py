import logging
import numpy as np
from .config import Config  # Assuming Config is defined in config.py
from typing import List, Dict, Tuple

logger = logging.getLogger("anomaly-detector")

class AnomalyDetector:
    def __init__(self, latency_percentile: int):
        self.latency_percentile = latency_percentile

    def _calculate_trace_duration(self, trace: Dict) -> float:
        """
        Calculate the total duration of a trace.
        
        Args:
            trace: Single trace object
            
        Returns:
            float: Total duration in microseconds
        """
        spans = trace.get("spans", [])
        if not spans:
            return 0.0
        
        # Method 1: Find the root span (span with no parent references)
        # This is typically the most accurate for end-to-end latency
        root_spans = []
        for span in spans:
            references = span.get("references", [])
            # A root span has no CHILD_OF references
            if not any(ref.get("refType") == "CHILD_OF" for ref in references):
                root_spans.append(span)
        
        if root_spans:
            # Use the longest root span duration
            return max(span.get("duration", 0) for span in root_spans)
        
        # Method 2: If no clear root span, use the span with the earliest start time
        # and latest end time to calculate total trace duration
        if spans:
            start_times = [span.get("startTime", 0) for span in spans]
            end_times = [span.get("startTime", 0) + span.get("duration", 0) for span in spans]
            
            if start_times and end_times:
                total_duration = max(end_times) - min(start_times)
                return total_duration
        
        # Method 3: Fallback - sum all span durations (may double-count)
        return sum(span.get("duration", 0) for span in spans)

    def _calculate_threshold(self, traces: List[Dict]) -> Tuple[float, List[float]]:
        """
        Calculate the latency threshold and return all latencies
        
        Args:
            traces: List of traces to analyze
            
        Returns:
            Tuple[float, List[float]]: Threshold value and list of all latencies
        """
        latencies = [self._calculate_trace_duration(trace) for trace in traces]
        threshold = np.percentile(latencies, self.latency_percentile)
        logger.info(f"Latency threshold ({self.latency_percentile}th percentile): {threshold:.2f} microseconds")
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
        
        # Filter traces that exceed the threshold using calculated trace duration
        anomalous_traces = []
        for trace in traces:
            trace_duration = self._calculate_trace_duration(trace)
            if trace_duration > threshold:
                anomalous_traces.append(trace)
        
        logger.info(f"Found {len(anomalous_traces)} anomalous traces out of {len(traces)} total traces (threshold: {threshold:.2f}μs)")
        return anomalous_traces
    
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
        
        # Filter traces that are below or at the threshold using calculated trace duration
        normal_traces = []
        for trace in traces:
            trace_duration = self._calculate_trace_duration(trace)
            if trace_duration <= threshold:
                normal_traces.append(trace)
        
        logger.info(f"Found {len(normal_traces)} normal traces out of {len(traces)} total traces")
        return normal_traces