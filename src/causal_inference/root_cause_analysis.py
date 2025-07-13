"""
Root Cause Analysis using Causal Inference

This module provides high-level functions for performing root cause analysis
on microservice latency data using causal inference techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime

from .data_preparation import LatencyDataProcessor
from .causal_model import CausalGraphBuilder, CausalModelTrainer

logger = logging.getLogger(__name__)


class RootCauseAnalyzer:
    """Complete root cause analysis pipeline using causal inference."""
    
    def __init__(self):
        self.data_processor = LatencyDataProcessor()
        self.graph_builder = CausalGraphBuilder()
        self.model_trainer = None
        self.normal_data = None
        self.anomalous_data = None
        self.results = {}
    
    def load_data(self, trace_file: str, dependency_file: str) -> None:
        """Load trace and dependency data."""
        logger.info("Loading trace and dependency data...")
        
        # Load traces and extract latencies
        self.data_processor.load_trace_data(trace_file)
        self.data_processor.load_dependency_graph(dependency_file)
        self.data_processor.extract_service_latencies()
        
        # Prepare normal and anomalous datasets
        self.normal_data = self.data_processor.get_normal_data()
        self.anomalous_data = self.data_processor.get_anomalous_data()
        
        logger.info(f"Loaded {len(self.normal_data)} normal and {len(self.anomalous_data)} anomalous traces")
        
        # Build causal graph
        self.graph_builder.load_dependencies(dependency_file)
        causal_graph = self.graph_builder.build_causal_graph()
        
        # Initialize model trainer
        self.model_trainer = CausalModelTrainer(causal_graph)
    
    def train_causal_model(self, auto_assign_mechanisms: bool = True) -> None:
        """Train the causal model on normal data."""
        if not self.model_trainer:
            raise ValueError("No model trainer initialized. Call load_data() first.")
        
        logger.info("Training causal model...")
        
        self.model_trainer.create_structural_causal_model()
        
        if not auto_assign_mechanisms:
            self.model_trainer.assign_causal_mechanisms(auto_assign=False)
        
        self.model_trainer.train_model(self.normal_data)
        
        logger.info("Causal model training completed")
    
    def analyze_single_outlier(self, target_service: str, 
                             outlier_sample: Optional[pd.DataFrame] = None,
                             num_bootstrap: int = 10) -> Dict[str, Any]:
        """
        Analyze a single outlier case.
        
        Args:
            target_service: Service experiencing the anomaly
            outlier_sample: Specific outlier sample (if None, uses first anomalous sample)
            num_bootstrap: Number of bootstrap samples for confidence intervals
            
        Returns:
            Dictionary with attribution results
        """
        if not self.model_trainer or not self.model_trainer.is_trained:
            raise ValueError("Model not trained. Call train_causal_model() first.")
        
        logger.info(f"Analyzing single outlier for service: {target_service}")
        
        # Use provided sample or first anomalous sample
        if outlier_sample is None:
            if len(self.anomalous_data) == 0:
                raise ValueError("No anomalous data available")
            outlier_sample = self.anomalous_data.head(1)
        
        # Perform attribution
        median_attribs, uncertainty_attribs = self.model_trainer.attribute_anomalies(
            outlier_sample, target_service, num_bootstrap
        )
        
        # Calculate outlier magnitude
        normal_mean = self.normal_data[target_service].mean()
        outlier_value = outlier_sample[target_service].iloc[0]
        outlier_magnitude = outlier_value - normal_mean
        
        result = {
            'type': 'single_outlier',
            'target_service': target_service,
            'outlier_magnitude': outlier_magnitude,
            'normal_mean': normal_mean,
            'outlier_value': outlier_value,
            'attributions': median_attribs,
            'uncertainties': uncertainty_attribs,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results['single_outlier'] = result
        logger.info(f"Single outlier analysis completed. Magnitude: {outlier_magnitude:.2f}ms")
        
        return result
    
    def analyze_distribution_change(self, target_service: str,
                                  num_bootstrap: int = 10) -> Dict[str, Any]:
        """
        Analyze permanent distribution changes.
        
        Args:
            target_service: Service experiencing distribution change
            num_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with distribution change analysis results
        """
        if not self.model_trainer or not self.model_trainer.is_trained:
            raise ValueError("Model not trained. Call train_causal_model() first.")
        
        if len(self.anomalous_data) == 0:
            raise ValueError("No anomalous data available for distribution change analysis")
        
        logger.info(f"Analyzing distribution change for service: {target_service}")
        
        # Perform distribution change analysis
        median_attribs, uncertainty_attribs = self.model_trainer.analyze_distribution_change(
            self.normal_data, self.anomalous_data, target_service, num_bootstrap
        )
        
        # Calculate distribution change magnitude
        normal_mean = self.normal_data[target_service].mean()
        anomalous_mean = self.anomalous_data[target_service].mean()
        change_magnitude = anomalous_mean - normal_mean
        
        result = {
            'type': 'distribution_change',
            'target_service': target_service,
            'change_magnitude': change_magnitude,
            'normal_mean': normal_mean,
            'anomalous_mean': anomalous_mean,
            'attributions': median_attribs,
            'uncertainties': uncertainty_attribs,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results['distribution_change'] = result
        logger.info(f"Distribution change analysis completed. Change: {change_magnitude:.2f}ms")
        
        return result
    
    def simulate_intervention(self, interventions: Dict[str, Any],
                            data_type: str = 'anomalous',
                            num_bootstrap: int = 10) -> Dict[str, Any]:
        """
        Simulate interventions to predict their effects.
        
        Args:
            interventions: Dictionary of {service: intervention_function}
            data_type: 'normal' or 'anomalous' data to apply interventions to
            num_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with intervention simulation results
        """
        if not self.model_trainer or not self.model_trainer.is_trained:
            raise ValueError("Model not trained. Call train_causal_model() first.")
        
        # Select data
        if data_type == 'normal':
            data = self.normal_data
        elif data_type == 'anomalous':
            data = self.anomalous_data
            if len(data) == 0:
                raise ValueError("No anomalous data available")
        else:
            raise ValueError("data_type must be 'normal' or 'anomalous'")
        
        logger.info(f"Simulating interventions on {data_type} data: {list(interventions.keys())}")
        
        # Perform intervention simulation
        median_results, uncertainty_results = self.model_trainer.simulate_intervention(
            data, interventions, num_bootstrap
        )
        
        # Calculate baseline means for comparison
        baseline_means = data.mean().to_dict()
        
        result = {
            'type': 'intervention_simulation',
            'data_type': data_type,
            'interventions': {k: str(v) for k, v in interventions.items()},  # Convert functions to strings
            'baseline_means': baseline_means,
            'intervention_results': median_results,
            'uncertainties': uncertainty_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results['intervention_simulation'] = result
        logger.info("Intervention simulation completed")
        
        return result
    
    def get_service_summary(self) -> Dict[str, Any]:
        """Get summary of services and their characteristics."""
        if not self.graph_builder.causal_graph:
            raise ValueError("No causal graph built. Call load_data() first.")
        
        graph = self.graph_builder.causal_graph
        
        summary = {
            'total_services': len(graph.nodes),
            'total_causal_edges': len(graph.edges),
            'root_services': self.graph_builder.get_root_nodes(),
            'leaf_services': self.graph_builder.get_leaf_nodes(),
            'service_latency_stats': {}
        }
        
        # Add latency statistics for each service
        if self.normal_data is not None:
            for service in self.normal_data.columns:
                stats = {
                    'mean': float(self.normal_data[service].mean()),
                    'std': float(self.normal_data[service].std()),
                    'min': float(self.normal_data[service].min()),
                    'max': float(self.normal_data[service].max())
                }
                summary['service_latency_stats'][service] = stats
        
        return summary
    
    def visualize_results(self, result_type: str, save_dir: Optional[str] = None) -> None:
        """
        Visualize analysis results.
        
        Args:
            result_type: Type of result to visualize ('single_outlier', 'distribution_change', etc.)
            save_dir: Directory to save plots (optional)
        """
        if result_type not in self.results:
            raise ValueError(f"No results available for type: {result_type}")
        
        result = self.results[result_type]
        
        # Create bar plot with uncertainty
        attributions = result['attributions']
        uncertainties = result['uncertainties']
        
        self._create_attribution_plot(attributions, uncertainties, result, save_dir)
    
    def _create_attribution_plot(self, attributions: Dict[str, float], 
                               uncertainties: Dict[str, Tuple[float, float]],
                               result: Dict[str, Any],
                               save_dir: Optional[str] = None) -> None:
        """Create attribution bar plot with uncertainty bars."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        services = list(attributions.keys())
        values = list(attributions.values())
        
        # Calculate error bars
        yerr_minus = [attributions[service] - uncertainties[service][0] for service in services]
        yerr_plus = [uncertainties[service][1] - attributions[service] for service in services]
        
        # Create bar plot
        bars = ax.bar(services, values, 
                     yerr=[yerr_minus, yerr_plus], 
                     ecolor='#1E88E5', color='#ff0d57', 
                     capsize=5, error_kw={'linewidth': 2})
        
        # Customize plot
        ax.set_ylabel('Attribution Score')
        ax.set_title(f'Root Cause Attribution - {result["target_service"]} '
                    f'({result["type"].replace("_", " ").title()})')
        ax.tick_params(axis='x', rotation=45)
        
        # Remove top and right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save if requested
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            filename = f"{result['type']}_{result['target_service']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path / filename}")
        
        plt.show()
    
    def save_results(self, output_dir: str) -> str:
        """Save all analysis results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rca_results_{timestamp}.json"
        filepath = output_path / filename
        
        # Add service summary to results
        results_with_summary = {
            'service_summary': self.get_service_summary(),
            'analyses': self.results,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'normal_samples': len(self.normal_data) if self.normal_data is not None else 0,
                'anomalous_samples': len(self.anomalous_data) if self.anomalous_data is not None else 0
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_with_summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
        return str(filepath)


def run_complete_rca_analysis(trace_file: str, dependency_file: str,
                            target_service: str, output_dir: str) -> RootCauseAnalyzer:
    """
    Run complete root cause analysis pipeline.
    
    Args:
        trace_file: Path to trace data JSON file
        dependency_file: Path to dependency analysis JSON file
        target_service: Service to analyze for anomalies
        output_dir: Directory to save results
        
    Returns:
        Configured RootCauseAnalyzer with completed analysis
    """
    analyzer = RootCauseAnalyzer()
    
    try:
        # Load data and train model
        analyzer.load_data(trace_file, dependency_file)
        analyzer.train_causal_model()
        
        # Perform analyses
        if len(analyzer.anomalous_data) > 0:
            analyzer.analyze_single_outlier(target_service)
            analyzer.analyze_distribution_change(target_service)
        else:
            logger.warning("No anomalous data found - skipping anomaly analyses")
        
        # Save results
        analyzer.save_results(output_dir)
        
        logger.info("Complete RCA analysis finished successfully")
        
    except Exception as e:
        logger.error(f"Error in RCA analysis: {e}")
        raise
    
    return analyzer


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    trace_file = "output/traces/collected_traces_20250713_230056.json"
    dependency_file = "output/analysis/dependency_analysis_20250713_230114.json"
    target_service = "frontend"  # Adjust based on your services
    output_dir = "output/rca_results"
    
    analyzer = run_complete_rca_analysis(trace_file, dependency_file, target_service, output_dir)
    print("RCA analysis completed successfully")
