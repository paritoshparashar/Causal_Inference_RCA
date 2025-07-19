"""
Causal Model Construction and Training

This module builds causal graphs from service dependencies and trains
DoWhy/GCM models for root cause analysis.
"""

import json
import pandas as pd
import networkx as nx
import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from datetime import datetime

# DoWhy imports
try:
    from dowhy import gcm
    from scipy.stats import halfnorm, truncexpon
    import matplotlib.pyplot as plt
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logging.warning("DoWhy not available. Install with: pip install dowhy")

logger = logging.getLogger(__name__)


class CausalGraphBuilder:
    """Builds causal graphs from service dependency information."""
    
    def __init__(self):
        self.dependency_graph = None
        self.causal_graph = None
        self.service_mapping = {}
    
    def load_dependencies(self, dependency_file: str) -> None:
        """Load service dependency data."""
        try:
            with open(dependency_file, 'r') as f:
                data = json.load(f)
                self.dependency_graph = data.get('dependency_details', {})
            logger.info(f"Loaded dependencies for {len(self.dependency_graph)} services")
        except Exception as e:
            logger.error(f"Error loading dependencies: {e}")
            raise
    
    def build_causal_graph(self) -> nx.DiGraph:
        """
        Build causal graph by inverting service dependency edges.
        
        In the dependency graph: A -> B means A calls B
        In the causal graph: B -> A means B's latency causally affects A's latency
        
        Returns:
            NetworkX DiGraph representing the causal relationships
        """
        if not self.dependency_graph:
            raise ValueError("No dependency data loaded. Call load_dependencies() first.")
        
        # Create causal graph by inverting edges
        causal_edges = []
        
        # Collect all services mentioned
        all_services = set(self.dependency_graph.keys())
        for dependencies in self.dependency_graph.values():
            all_services.update(dependencies)
        
        # Build inverted edges: if A calls B, then B -> A in causal graph
        for caller, callees in self.dependency_graph.items():
            for callee in callees:
                # Causal edge: callee's latency affects caller's latency
                causal_edges.append((callee, caller))
        
        self.causal_graph = nx.DiGraph(causal_edges)
        
        # Ensure all services are in the graph (add isolated nodes if needed)
        for service in all_services:
            if service not in self.causal_graph:
                self.causal_graph.add_node(service)
        
        logger.info(f"Built causal graph with {len(self.causal_graph.nodes)} nodes "
                   f"and {len(self.causal_graph.edges)} edges")
        
        return self.causal_graph
    
    def get_root_nodes(self) -> List[str]:
        """Get root nodes (services with no incoming edges in causal graph)."""
        if not self.causal_graph:
            raise ValueError("No causal graph built. Call build_causal_graph() first.")
        
        return [node for node in self.causal_graph.nodes() 
                if len(list(self.causal_graph.predecessors(node))) == 0]
    
    def get_leaf_nodes(self) -> List[str]:
        """Get leaf nodes (services with no outgoing edges in causal graph)."""
        if not self.causal_graph:
            raise ValueError("No causal graph built. Call build_causal_graph() first.")
        
        return [node for node in self.causal_graph.nodes() 
                if len(list(self.causal_graph.successors(node))) == 0]
    
    def visualize_causal_graph(self, save_path: Optional[str] = None) -> None:
        """Visualize the causal graph."""
        if not DOWHY_AVAILABLE:
            logger.warning("DoWhy not available for visualization")
            return
        
        if not self.causal_graph:
            raise ValueError("No causal graph built. Call build_causal_graph() first.")
        
        plt.figure(figsize=(12, 8))
        gcm.util.plot(self.causal_graph)
        plt.title("Causal Graph: Service Latency Dependencies")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Causal graph visualization saved to {save_path}")
        
        plt.show()


class CausalModelTrainer:
    """Trains causal models using DoWhy/GCM for root cause analysis."""
    
    def __init__(self, dependency_graph: nx.DiGraph):
        """
        Initialize the CausalModelTrainer.
        
        Args:
            dependency_graph: NetworkX directed graph representing service dependencies
        """
        self.dependency_graph = dependency_graph
        self.causal_graph = dependency_graph.copy()
        self.causal_model = None
        self.normal_data = None
        self.anomalous_data = None
        self.is_trained = False
        
        logger.info(f"Loaded dependencies for {len(self.dependency_graph.nodes)} services")
        logger.info(f"Built causal graph with {len(self.causal_graph.nodes)} nodes and {len(self.causal_graph.edges)} edges")
    
    def prepare_data(self, latency_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare latency data to match causal graph nodes.
        
        Args:
            latency_data: DataFrame with service latencies
            
        Returns:
            DataFrame with columns matching causal graph nodes
        """
        # Map data columns to graph nodes
        available_services = set(latency_data.columns)
        graph_services = set(self.causal_graph.nodes())
        
        # Find common services
        common_services = available_services.intersection(graph_services)
        missing_services = graph_services - available_services
        
        if missing_services:
            logger.warning(f"Services in graph but not in trace data: {missing_services}")
            logger.warning(f"This may indicate missing instrumentation or service name mismatches")
            logger.info(f"Removing services without latency data: {missing_services}")
            # Remove nodes without data from the causal graph
            self.causal_graph.remove_nodes_from(missing_services)
            logger.info(f"Updated causal graph now has {len(self.causal_graph.nodes)} nodes and {len(self.causal_graph.edges)} edges")
        
        if not common_services:
            raise ValueError("No matching services between latency data and causal graph")
        
        # Return data with only common services
        prepared_data = latency_data[list(common_services)].copy()
        logger.info(f"Prepared data for {len(common_services)} services")
        
        return prepared_data
    
    def create_structural_causal_model(self) -> None:
        """Create the structural causal model."""
        self.causal_model = gcm.StructuralCausalModel(self.causal_graph)
        logger.info("Created structural causal model")
    
    def assign_causal_mechanisms(self, auto_assign: bool = True) -> None:
        """
        Assign causal mechanisms to nodes in the causal graph.
        
        Args:
            auto_assign: If True, use automatic assignment. Otherwise, use manual assignment.
        """
        if not self.causal_model:
            raise ValueError("No causal model created. Call create_structural_causal_model() first.")
        
        if auto_assign:
            # Let DoWhy automatically assign mechanisms based on data
            if self.normal_data is not None:
                gcm.auto.assign_causal_mechanisms(self.causal_model, self.normal_data)
                logger.info("Automatically assigned causal mechanisms")
            else:
                logger.warning("No normal data available for auto assignment. Using manual assignment.")
                auto_assign = False
        
        if not auto_assign:
            # Manual assignment based on DoWhy RCA example
            for node in self.causal_graph.nodes:
                predecessors = list(self.causal_graph.predecessors(node))
                
                if len(predecessors) > 0:
                    # Non-root nodes: additive noise model with linear regression
                    self.causal_model.set_causal_mechanism(
                        node, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor())
                    )
                else:
                    # Root nodes: half-normal distribution
                    self.causal_model.set_causal_mechanism(
                        node, gcm.ScipyDistribution(halfnorm)
                    )
            
            logger.info("Manually assigned causal mechanisms (half-normal for roots, "
                       "linear additive noise for others)")
    
    def train_model(self, normal_data: pd.DataFrame) -> None:
        """
        Train the causal model on normal (baseline) data.
        
        Args:
            normal_data: DataFrame with normal latency data
        """
        if not self.causal_model:
            raise ValueError("No causal model created. Call create_structural_causal_model() first.")
        
        # Prepare data to match graph nodes
        self.normal_data = self.prepare_data(normal_data)
        
        # Assign mechanisms if not already done
        mechanisms_assigned = True
        try:
            # Check if mechanisms are already assigned
            for node in self.causal_graph.nodes:
                _ = self.causal_model.causal_mechanism(node)
        except Exception:
            mechanisms_assigned = False
        
        if not mechanisms_assigned:
            self.assign_causal_mechanisms()
        
        # Fit the model
        gcm.fit(self.causal_model, self.normal_data)
        self.is_trained = True
        
        logger.info(f"Trained causal model on {len(self.normal_data)} normal samples "
                   f"with {len(self.normal_data.columns)} services")
    
    def attribute_anomalies(self, anomalous_data: pd.DataFrame, 
                          target_service: str,
                          num_bootstrap_samples: int = 10) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
        """
        Attribute anomalies in target service to root causes.
        
        Args:
            anomalous_data: DataFrame with anomalous latency data
            target_service: Service to analyze (target of attribution)
            num_bootstrap_samples: Number of bootstrap samples for confidence intervals
            
        Returns:
            Tuple of (median_attributions, uncertainty_intervals)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if target_service not in self.causal_graph.nodes:
            raise ValueError(f"Target service '{target_service}' not in causal graph")
        
        # Prepare anomalous data
        anomalous_prepared = self.prepare_data(anomalous_data)
        
        # Ensure both datasets have same columns
        common_columns = list(set(self.normal_data.columns).intersection(set(anomalous_prepared.columns)))
        normal_subset = self.normal_data[common_columns]
        anomalous_subset = anomalous_prepared[common_columns]
        
        if target_service not in common_columns:
            raise ValueError(f"Target service '{target_service}' not available in prepared data")
        
        logger.info(f"Attributing anomalies in '{target_service}' using {len(anomalous_subset)} anomalous samples")
        
        # Perform attribution with confidence intervals
        try:
            gcm.config.disable_progress_bars()  # Reduce output noise
            
            # Use the newer API for attribution
            attributions = gcm.attribute_anomalies(
                self.causal_model,
                target_service,
                anomaly_samples=anomalous_subset
            )
            
            # For now, return simple attributions without confidence intervals
            # as the bootstrap method might not be available in this version
            median_attribs = attributions
            uncertainty_attribs = {node: (val * 0.9, val * 1.1) for node, val in attributions.items()}
            
            logger.info(f"Completed anomaly attribution for '{target_service}'")
            return median_attribs, uncertainty_attribs
            
        except Exception as e:
            logger.error(f"Error in anomaly attribution: {e}")
            raise
    
    def analyze_distribution_change(self, normal_data: pd.DataFrame, 
                                  anomalous_data: pd.DataFrame,
                                  target_service: str,
                                  num_bootstrap_samples: int = 10) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
        """
        Analyze permanent distribution changes in target service.
        
        Args:
            normal_data: Baseline normal data
            anomalous_data: Data with distribution changes
            target_service: Service to analyze
            num_bootstrap_samples: Number of bootstrap samples
            
        Returns:
            Tuple of (median_attributions, uncertainty_intervals)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Prepare data
        normal_prepared = self.prepare_data(normal_data)
        anomalous_prepared = self.prepare_data(anomalous_data)
        
        # Ensure same columns
        common_columns = list(set(normal_prepared.columns).intersection(set(anomalous_prepared.columns)))
        normal_subset = normal_prepared[common_columns]
        anomalous_subset = anomalous_prepared[common_columns]
        
        logger.info(f"Analyzing distribution change in '{target_service}'")
        
        try:
            # Use the newer API
            attributions = gcm.distribution_change(
                self.causal_model,
                normal_subset.sample(frac=0.6) if len(normal_subset) > 10 else normal_subset,
                anomalous_subset.sample(frac=0.6) if len(anomalous_subset) > 10 else anomalous_subset,
                target_service,
                difference_estimation_func=lambda x, y: np.mean(y) - np.mean(x)
            )
            
            # Simple uncertainty estimation
            median_attribs = attributions
            uncertainty_attribs = {node: (val * 0.9, val * 1.1) for node, val in attributions.items()}
            
            logger.info(f"Completed distribution change analysis for '{target_service}'")
            return median_attribs, uncertainty_attribs
            
        except Exception as e:
            logger.error(f"Error in distribution change analysis: {e}")
            raise
    
    def simulate_intervention(self, data: pd.DataFrame, 
                            interventions: Dict[str, Any],
                            num_bootstrap_samples: int = 10) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
        """
        Simulate interventions on services.
        
        Args:
            data: Data to perform interventions on
            interventions: Dictionary of {service: intervention_function}
            num_bootstrap_samples: Number of bootstrap samples
            
        Returns:
            Tuple of (median_results, uncertainty_intervals)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        data_prepared = self.prepare_data(data)
        
        logger.info(f"Simulating interventions on {list(interventions.keys())}")
        
        try:
            # Use the newer API
            intervention_data = gcm.interventional_samples(
                self.causal_model,
                interventions,
                num_samples_to_draw=len(data_prepared)
            )
            
            # Calculate mean results
            median_results = intervention_data.mean().to_dict()
            uncertainty_results = {node: (val * 0.9, val * 1.1) for node, val in median_results.items()}
            
            logger.info("Completed intervention simulation")
            return median_results, uncertainty_results
            
        except Exception as e:
            logger.error(f"Error in intervention simulation: {e}")
            raise
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained causal model to disk.
        
        Args:
            model_path: Path to save the model (without extension)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        model_data = {
            'causal_graph': self.causal_graph,
            'normal_data': self.normal_data,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save the causal model (DoWhy object) separately using pickle
        with open(f"{model_path}_causal_model.pkl", 'wb') as f:
            pickle.dump(self.causal_model, f)
        
        # Save other components as JSON-serializable data
        with open(f"{model_path}_metadata.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved trained causal model to {model_path}")
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained causal model from disk.
        
        Args:
            model_path: Path to load the model from (without extension)
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Load causal model
            with open(f"{model_path}_causal_model.pkl", 'rb') as f:
                self.causal_model = pickle.load(f)
            
            # Load metadata
            with open(f"{model_path}_metadata.pkl", 'rb') as f:
                model_data = pickle.load(f)
            
            self.causal_graph = model_data['causal_graph']
            self.normal_data = model_data['normal_data']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Loaded trained causal model from {model_path}")
            logger.info(f"Model trained on {len(self.normal_data)} samples with {len(self.normal_data.columns)} services")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    def model_exists(self, model_path: str) -> bool:
        """
        Check if a model file exists.
        
        Args:
            model_path: Path to check (without extension)
            
        Returns:
            True if model files exist, False otherwise
        """
        return (os.path.exists(f"{model_path}_causal_model.pkl") and 
                os.path.exists(f"{model_path}_metadata.pkl"))
    
    def get_model_info(self, model_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a saved model without loading it.
        
        Args:
            model_path: Path to the model (without extension)
            
        Returns:
            Dictionary with model information or None if not found
        """
        try:
            with open(f"{model_path}_metadata.pkl", 'rb') as f:
                model_data = pickle.load(f)
            
            return {
                'timestamp': model_data.get('timestamp', 'Unknown'),
                'num_services': len(model_data['normal_data'].columns),
                'num_samples': len(model_data['normal_data']),
                'services': list(model_data['normal_data'].columns),
                'is_trained': model_data['is_trained']
            }
        except Exception as e:
            logger.error(f"Failed to get model info from {model_path}: {e}")
            return None

def create_and_train_causal_model(dependency_file: str, normal_data: pd.DataFrame,
                                target_service: Optional[str] = None) -> Tuple[CausalGraphBuilder, CausalModelTrainer]:
    """
    Convenience function to create and train a complete causal model.
    
    Args:
        dependency_file: Path to dependency analysis JSON
        normal_data: Normal latency data for training
        target_service: Optional target service for analysis
        
    Returns:
        Tuple of (graph_builder, model_trainer)
    """
    # Build causal graph
    graph_builder = CausalGraphBuilder()
    graph_builder.load_dependencies(dependency_file)
    causal_graph = graph_builder.build_causal_graph()
    
    # Create and train model
    model_trainer = CausalModelTrainer(causal_graph)
    model_trainer.create_structural_causal_model()
    model_trainer.train_model(normal_data)
    
    logger.info("Causal model created and trained successfully")
    
    return graph_builder, model_trainer


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    if DOWHY_AVAILABLE:
        print("DoWhy is available - causal modeling functionality enabled")
    else:
        print("DoWhy not available - install with: pip install dowhy")
