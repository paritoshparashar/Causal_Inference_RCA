#!/usr/bin/env python3
"""
Training Pipeline - Handles model training operations
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from trace_collection import Config, AnomalyDetector
from dependency_analysis import DependencyGraphBuilder

try:
    from causal_inference.root_cause_analysis import RootCauseAnalyzer
    CAUSAL_INFERENCE_AVAILABLE = True
except ImportError:
    CAUSAL_INFERENCE_AVAILABLE = False

logger = logging.getLogger("training_pipeline")


class TrainingPipeline:
    """Handles model training operations"""
    
    def __init__(self):
        self.config = Config()
    
    def train_from_file(self, traces_file: str) -> dict:
        """
        Train model using existing trace file.
        
        Returns:
            dict with success status and file paths
        """
        logger.info(f"Loading traces from file: {traces_file}")
        
        if not os.path.exists(traces_file):
            raise FileNotFoundError(f"Trace file not found: {traces_file}")
        
        # Load and process traces
        traces = self._load_traces(traces_file)
        logger.info(f"Processing {len(traces)} traces from file")
        
        # Detect anomalies
        anomaly_detector = AnomalyDetector(self.config.LATENCY_PERCENTILE)
        logger.info("Detecting anomalous traces...")
        anomalous_traces = anomaly_detector.detect(traces)
        logger.info(f"Found {len(anomalous_traces)} anomalous traces")
        
        # Save anomalous traces
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        anomalous_file = self._save_anomalous_traces(anomalous_traces, timestamp)
        
        # Generate dependency analysis
        logger.info("Analyzing dependencies and training causal model...")
        dependency_file = self._analyze_dependencies(traces_file, timestamp)
        
        # Train causal model
        model_path = self._train_causal_model(traces_file, dependency_file)
        
        return {
            'success': True,
            'traces_file': traces_file,
            'anomalous_file': anomalous_file,
            'dependency_file': dependency_file,
            'model_path': model_path
        }
    
    def _load_traces(self, traces_file: str) -> list:
        """Load traces from file handling different formats"""
        with open(traces_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'data' in data:
            return data['data']
        elif isinstance(data, list):
            return data
        else:
            return [data]
    
    def _save_anomalous_traces(self, anomalous_traces: list, timestamp: str) -> str:
        """Save anomalous traces to file"""
        output_dir = Path("output/traces")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        anomalous_file = output_dir / f"anomalous_traces_{timestamp}.json"
        with open(anomalous_file, 'w') as f:
            json.dump(anomalous_traces, f, indent=2)
        
        logger.info(f"Saved anomalous traces: {anomalous_file}")
        return str(anomalous_file)
    
    def _analyze_dependencies(self, traces_file: str, timestamp: str) -> str:
        """Perform dependency analysis"""
        from ..dependency_analysis.analyze_dependencies import run_dependency_analysis
        
        output_files = run_dependency_analysis(traces_file, timestamp)
        return output_files[0]  # Return JSON summary file
    
    def _train_causal_model(self, traces_file: str, dependency_file: str) -> str:
        """Train the causal model"""
        if not CAUSAL_INFERENCE_AVAILABLE:
            raise ImportError("Causal inference modules not available")
        
        analyzer = RootCauseAnalyzer()
        analyzer.load_data(traces_file, dependency_file, use_cache=False)
        analyzer.train_causal_model(dependency_file=dependency_file)
        return analyzer.get_model_path(dependency_file)
