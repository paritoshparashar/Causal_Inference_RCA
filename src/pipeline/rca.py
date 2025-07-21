#!/usr/bin/env python3
"""
RCA Pipeline - Handles root cause analysis operations
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from trace_collection import Config, AnomalyDetector

try:
    from causal_inference.root_cause_analysis import RootCauseAnalyzer
    CAUSAL_INFERENCE_AVAILABLE = True
except ImportError:
    CAUSAL_INFERENCE_AVAILABLE = False

logger = logging.getLogger("rca_pipeline")


class RCAPipeline:
    """Handles RCA analysis operations"""
    
    def __init__(self):
        self.config = Config()
    
    def analyze_from_file(self, traces_file: str, model_path: str, target_service: str) -> dict:
        """
        Perform RCA on existing trace file.
        
        Returns:
            dict with success status and result file path
        """
        logger.info(f"RCA Analysis - Target: {target_service}")
        logger.info(f"Traces file: {traces_file}")
        logger.info(f"Model path: {model_path}")
        
        # Validate inputs
        self._validate_inputs(traces_file, model_path)
        
        # Load and process traces
        traces = self._load_traces(traces_file)
        logger.info(f"Processing {len(traces)} traces")
        
        # Detect anomalies
        anomalous_traces = self._detect_anomalies(traces)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        anomalous_file = self._save_anomalous_traces(anomalous_traces, timestamp)
        
        # Find or generate dependency file
        dependency_file = self._get_dependency_file(traces_file, timestamp)
        
        # Perform RCA
        rca_file = self._run_causal_inference(traces_file, dependency_file, target_service, timestamp)
        
        return {
            'success': True,
            'traces_file': traces_file,
            'anomalous_file': anomalous_file,
            'rca_file': rca_file
        }
    
    def _validate_inputs(self, traces_file: str, model_path: str):
        """Validate input files exist"""
        if not os.path.exists(traces_file):
            raise FileNotFoundError(f"Trace file not found: {traces_file}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def _load_traces(self, traces_file: str) -> list:
        """Load traces from file"""
        with open(traces_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'data' in data:
            return data['data']
        elif isinstance(data, list):
            return data
        else:
            return [data]
    
    def _detect_anomalies(self, traces: list) -> list:
        """Detect anomalous traces"""
        anomaly_detector = AnomalyDetector(self.config.LATENCY_PERCENTILE)
        logger.info("Detecting anomalous traces...")
        anomalous_traces = anomaly_detector.detect(traces)
        logger.info(f"Found {len(anomalous_traces)} anomalous traces")
        return anomalous_traces
    
    def _save_anomalous_traces(self, anomalous_traces: list, timestamp: str) -> str:
        """Save anomalous traces to file"""
        output_dir = Path("output/traces")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        anomalous_file = output_dir / f"anomalous_traces_{timestamp}.json"
        with open(anomalous_file, 'w') as f:
            json.dump(anomalous_traces, f, indent=2)
        
        return str(anomalous_file)
    
    def _get_dependency_file(self, traces_file: str, timestamp: str) -> str:
        """Find existing dependency file or generate new one"""
        dependency_dir = Path("output/analysis")
        
        if dependency_dir.exists():
            for file_path in dependency_dir.glob("dependency_analysis_*.json"):
                return str(file_path)
        
        logger.warning("No existing dependency file found. Generating one...")
        from ..dependency_analysis.analyze_dependencies import run_dependency_analysis
        output_files = run_dependency_analysis(traces_file, timestamp)
        return output_files[0]
    
    def _run_causal_inference(self, traces_file: str, dependency_file: str, 
                             target_service: str, timestamp: str) -> Optional[str]:
        """Perform causal inference analysis"""
        if not CAUSAL_INFERENCE_AVAILABLE:
            logger.warning("Causal inference not available - skipping RCA")
            return None
        
        logger.info("=== CAUSAL INFERENCE RCA ===")
        
        try:
            analyzer = RootCauseAnalyzer()
            analyzer.load_data(traces_file, dependency_file, use_cache=True)
            analyzer.train_causal_model(dependency_file=dependency_file)
            
            # Get available services
            available_services = list(analyzer.normal_data.columns) if analyzer.normal_data is not None else []
            
            # Check target service exists
            if target_service not in available_services:
                logger.warning(f"Target service '{target_service}' not found in data.")
                if available_services:
                    target_service = available_services[0]
                    logger.info(f"Using '{target_service}' as target service instead")
                else:
                    logger.warning("No services available for RCA")
                    return None
            
            # Perform analyses
            results = {}
            if len(analyzer.anomalous_data) > 0:
                logger.info(f"Performing RCA for service: {target_service}")
                
                outlier_result = analyzer.analyze_single_outlier(target_service, num_bootstrap=5)
                results['single_outlier'] = outlier_result
                
                dist_result = analyzer.analyze_distribution_change(target_service, num_bootstrap=5)
                results['distribution_change'] = dist_result
                
                logger.info("RCA completed")
            else:
                logger.warning("No anomalous data found - skipping anomaly-specific RCA")
            
            # Add metadata
            results['service_summary'] = analyzer.get_service_summary()
            results['metadata'] = {
                'target_service': target_service,
                'timestamp': timestamp,
                'normal_samples': len(analyzer.normal_data) if analyzer.normal_data is not None else 0,
                'anomalous_samples': len(analyzer.anomalous_data) if analyzer.anomalous_data is not None else 0,
                'model_cached': True,
                'cached_models': analyzer.list_cached_models()
            }
            
            # Save results
            output_dir = Path("output/rca")
            output_dir.mkdir(parents=True, exist_ok=True)
            rca_file = output_dir / f"rca_results_{target_service}_{timestamp}.json"
            
            with open(rca_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"RCA results saved: {rca_file}")
            return str(rca_file)
            
        except Exception as e:
            logger.error(f"Causal inference RCA failed: {e}")
            return None
