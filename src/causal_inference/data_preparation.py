"""
Data Preparation for Causal Inference

This module extracts and prepares latency data from collected traces
for use in causal modeling and root cause analysis.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LatencyDataProcessor:
    """Processes trace data to extract service latency information for causal modeling."""
    
    def __init__(self):
        self.trace_data = None
        self.service_latencies = None
        self.dependency_graph = None
    
    def load_trace_data(self, trace_file_path: str) -> None:
        """Load trace data from JSON file."""
        try:
            with open(trace_file_path, 'r') as f:
                self.trace_data = json.load(f)
            logger.info(f"Loaded {len(self.trace_data)} traces from {trace_file_path}")
        except Exception as e:
            logger.error(f"Error loading trace data: {e}")
            raise
    
    def load_dependency_graph(self, dependency_file_path: str) -> None:
        """Load dependency graph data."""
        try:
            with open(dependency_file_path, 'r') as f:
                data = json.load(f)
                self.dependency_graph = data.get('dependency_details', {})
            logger.info(f"Loaded dependency graph with {len(self.dependency_graph)} services")
        except Exception as e:
            logger.error(f"Error loading dependency graph: {e}")
            raise
    
    def extract_service_latencies(self) -> pd.DataFrame:
        """
        Extract service latencies from trace data.
        
        Returns:
            DataFrame with columns for each service containing latency values per trace
        """
        if not self.trace_data:
            raise ValueError("No trace data loaded. Call load_trace_data() first.")
        
        # Dictionary to store latencies per service per trace
        latency_data = {}
        
        for trace in self.trace_data:
            trace_id = trace['traceID']
            
            # Extract latencies for each span/service in this trace
            trace_latencies = {}
            
            for span in trace['spans']:
                service_name = self._extract_service_name(span['operationName'])
                duration_us = span.get('duration', 0)  # Duration in microseconds
                duration_ms = duration_us / 1000.0  # Convert to milliseconds
                
                # If multiple spans for same service in trace, sum the durations
                if service_name in trace_latencies:
                    trace_latencies[service_name] += duration_ms
                else:
                    trace_latencies[service_name] = duration_ms
            
            # Store this trace's latencies
            for service, latency in trace_latencies.items():
                if service not in latency_data:
                    latency_data[service] = []
                latency_data[service].append(latency)
        
        # Convert to DataFrame, filling missing values with NaN
        max_length = max(len(latencies) for latencies in latency_data.values())
        
        # Pad shorter lists with NaN
        for service in latency_data:
            while len(latency_data[service]) < max_length:
                latency_data[service].append(np.nan)
        
        self.service_latencies = pd.DataFrame(latency_data)
        
        # Drop rows with any NaN values to ensure complete data
        self.service_latencies = self.service_latencies.dropna()
        
        logger.info(f"Extracted latencies for {len(self.service_latencies.columns)} services "
                   f"across {len(self.service_latencies)} complete traces")
        
        return self.service_latencies
    
    def _extract_service_name(self, operation_name: str) -> str:
        """Extract service name from operation name."""
        # Handle different operation name patterns
        if 'Server_' in operation_name:
            # Pattern: ServiceServer_Operation -> Service
            return operation_name.split('Server_')[0].replace('Service', '').lower()
        elif 'Client_' in operation_name:
            # Pattern: ServiceClient_Operation -> Service
            return operation_name.split('Client_')[0].replace('Service', '').lower()
        elif 'Service_' in operation_name:
            # Pattern: serviceNameService_Operation -> serviceName
            return operation_name.split('Service_')[0].lower()
        else:
            # Fallback: use the operation name as is, cleaned up
            return operation_name.lower().replace('service', '')
    
    def get_normal_data(self, percentile_threshold: float = 95.0) -> pd.DataFrame:
        """
        Get 'normal' data by filtering out high-latency traces.
        
        Args:
            percentile_threshold: Traces with total latency above this percentile are excluded
            
        Returns:
            DataFrame with normal (non-anomalous) latency data
        """
        if self.service_latencies is None:
            raise ValueError("No service latencies extracted. Call extract_service_latencies() first.")
        
        # Calculate total latency per trace (sum across all services)
        total_latencies = self.service_latencies.sum(axis=1)
        
        # Find threshold for normal data
        threshold = np.percentile(total_latencies, percentile_threshold)
        
        # Filter normal data
        normal_mask = total_latencies <= threshold
        normal_data = self.service_latencies[normal_mask].copy()
        
        logger.info(f"Filtered normal data: {len(normal_data)} traces out of {len(self.service_latencies)} "
                   f"(threshold: {threshold:.2f}ms total latency)")
        
        return normal_data
    
    def get_anomalous_data(self, percentile_threshold: float = 95.0) -> pd.DataFrame:
        """
        Get anomalous data (high-latency traces).
        
        Args:
            percentile_threshold: Traces with total latency above this percentile are included
            
        Returns:
            DataFrame with anomalous latency data
        """
        if self.service_latencies is None:
            raise ValueError("No service latencies extracted. Call extract_service_latencies() first.")
        
        # Calculate total latency per trace
        total_latencies = self.service_latencies.sum(axis=1)
        
        # Find threshold for anomalous data
        threshold = np.percentile(total_latencies, percentile_threshold)
        
        # Filter anomalous data
        anomalous_mask = total_latencies > threshold
        anomalous_data = self.service_latencies[anomalous_mask].copy()
        
        logger.info(f"Filtered anomalous data: {len(anomalous_data)} traces out of {len(self.service_latencies)} "
                   f"(threshold: {threshold:.2f}ms total latency)")
        
        return anomalous_data
    
    def save_processed_data(self, output_dir: str, timestamp: str = None) -> Dict[str, str]:
        """
        Save processed latency data to CSV files.
        
        Args:
            output_dir: Directory to save files
            timestamp: Optional timestamp for file naming
            
        Returns:
            Dictionary with paths to saved files
        """
        if self.service_latencies is None:
            raise ValueError("No service latencies to save. Call extract_service_latencies() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files_saved = {}
        
        # Save complete latency data
        complete_file = output_path / f"service_latencies_{timestamp}.csv"
        self.service_latencies.to_csv(complete_file, index=False)
        files_saved['complete'] = str(complete_file)
        
        # Save normal data
        normal_data = self.get_normal_data()
        normal_file = output_path / f"normal_latencies_{timestamp}.csv"
        normal_data.to_csv(normal_file, index=False)
        files_saved['normal'] = str(normal_file)
        
        # Save anomalous data
        anomalous_data = self.get_anomalous_data()
        if len(anomalous_data) > 0:
            anomalous_file = output_path / f"anomalous_latencies_{timestamp}.csv"
            anomalous_data.to_csv(anomalous_file, index=False)
            files_saved['anomalous'] = str(anomalous_file)
        
        logger.info(f"Saved processed latency data to {output_dir}")
        return files_saved


def process_traces_for_causal_analysis(trace_file: str, dependency_file: str, 
                                     output_dir: str) -> Dict[str, str]:
    """
    Convenience function to process traces and prepare data for causal analysis.
    
    Args:
        trace_file: Path to trace JSON file
        dependency_file: Path to dependency analysis JSON file
        output_dir: Directory to save processed data
        
    Returns:
        Dictionary with paths to saved processed files
    """
    processor = LatencyDataProcessor()
    
    # Load data
    processor.load_trace_data(trace_file)
    processor.load_dependency_graph(dependency_file)
    
    # Extract latencies
    processor.extract_service_latencies()
    
    # Save processed data
    return processor.save_processed_data(output_dir)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    trace_file = "output/traces/collected_traces_20250713_230056.json"
    dependency_file = "output/analysis/dependency_analysis_20250713_230114.json"
    output_dir = "output/causal_data"
    
    files_saved = process_traces_for_causal_analysis(trace_file, dependency_file, output_dir)
    print("Processed files:", files_saved)
