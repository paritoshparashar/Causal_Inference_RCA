"""
Latency Calculator

Handles latency calculations, propagation, and anomaly injection.
"""

import random
from typing import Dict, List, Tuple
from .service_topology import ServiceTopology


class LatencyCalculator:
    """
    Calculates service latencies with anomaly injection and upstream propagation.
    
    Handles:
    - Normal baseline latencies
    - Anomaly injection with different severity levels
    - Upstream latency propagation (slow child affects parent)
    """
    
    # Anomaly severity distribution (10% total anomalous traces)
    ANOMALY_DISTRIBUTION = {
        1.5: 0.025,  # 2.5% of all traces: 1.5x slower
        1.75: 0.025, # 2.5% of all traces: 1.75x slower
        2.0: 0.025,  # 2.5% of all traces: 2x slower
        2.5: 0.025   # 2.5% of all traces: 2.5x slower
    }
    
    def __init__(self, topology: ServiceTopology):
        self.topology = topology
        
    def calculate_trace_latencies(self, slow_service: str = None, trace_type: str = 'normal') -> Dict[str, float]:
        """
        Calculate latencies for all services in a single trace.
        
        Args:
            slow_service: Service to slow down (if anomalous trace)
            trace_type: 'normal' or 'anomalous'
            
        Returns:
            Dict mapping service_name -> latency_ms
        """
        if trace_type == 'normal':
            return self._calculate_normal_latencies()
        elif trace_type == 'anomalous':
            if not slow_service:
                raise ValueError("slow_service must be specified for anomalous traces")
            return self._calculate_anomalous_latencies(slow_service)
        else:
            raise ValueError(f"Invalid trace_type: {trace_type}")
    
    def _calculate_normal_latencies(self) -> Dict[str, float]:
        """Calculate normal baseline latencies with small random variation."""
        latencies = {}
        
        for service_name in self.topology.get_service_names():
            config = self.topology.get_service_config(service_name)
            # Add small random jitter (±10%)
            jitter = random.uniform(0.9, 1.1)
            latencies[service_name] = config.baseline_latency_ms * jitter
            
        return latencies
    
    def _calculate_anomalous_latencies(self, slow_service: str) -> Dict[str, float]:
        """Calculate latencies with anomaly injection and upstream propagation."""
        if not self.topology.validate_service_name(slow_service):
            raise ValueError(f"Invalid service name: {slow_service}")
        
        # Start with normal latencies
        latencies = self._calculate_normal_latencies()
        
        # Choose anomaly severity based on distribution
        slowdown_factor = self._choose_anomaly_severity()
        
        # Apply slowdown to target service
        original_latency = latencies[slow_service]
        latencies[slow_service] = original_latency * slowdown_factor
        
        # Propagate latency increase upstream
        self._propagate_latency_upstream(latencies, slow_service, original_latency, slowdown_factor)
        
        return latencies
    
    def _choose_anomaly_severity(self) -> float:
        """Choose anomaly severity based on the specified distribution."""
        # Create weighted choices based on ANOMALY_DISTRIBUTION
        factors = list(self.ANOMALY_DISTRIBUTION.keys())
        weights = [self.ANOMALY_DISTRIBUTION[f] for f in factors]
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return random.choices(factors, weights=weights)[0]
    
    def _propagate_latency_upstream(self, latencies: Dict[str, float], slow_service: str, 
                                  original_latency: float, slowdown_factor: float):
        """
        Propagate latency increase upstream through the call chain.
        
        Logic: If a service becomes slower, all services that call it (directly or indirectly)
        will also become slower as they wait for the response.
        """
        latency_increase = (slowdown_factor - 1.0) * original_latency
        
        # Get all upstream services that will be affected
        upstream_services = self.topology.get_upstream_services(slow_service)
        
        for upstream_service in upstream_services:
            # Calculate propagation factor based on dependency depth
            propagation_factor = self._calculate_propagation_factor(upstream_service, slow_service)
            
            # Add propagated latency increase
            latencies[upstream_service] += latency_increase * propagation_factor
    
    def _calculate_propagation_factor(self, upstream_service: str, slow_service: str) -> float:
        """
        Calculate how much of the downstream latency increase affects the upstream service.
        
        Factors considered:
        - Direct vs indirect dependency
        - Parallel vs sequential calls
        - Service's own processing overhead
        """
        downstream_services = self.topology.get_downstream_services(upstream_service)
        
        if slow_service in downstream_services:
            # Direct dependency - full impact for sequential calls
            if len(downstream_services) == 1:
                return 1.0  # Sequential call - full latency impact
            else:
                # Parallel calls - impact depends on whether this is the slowest call
                return 0.8  # Slightly reduced impact for parallel calls
        else:
            # Indirect dependency - reduced impact
            return 0.6  # Cascading effect with some dampening
    
    def determine_trace_type(self, trace_index: int, total_traces: int) -> str:
        """
        Determine if a trace should be normal or anomalous based on the distribution.
        
        90% normal, 10% anomalous
        """
        anomalous_threshold = 0.9  # 90% normal traces
        
        # Use deterministic approach for consistent distribution
        normalized_position = trace_index / total_traces
        
        if normalized_position < anomalous_threshold:
            return 'normal'
        else:
            return 'anomalous'
    
    def get_anomaly_stats(self, total_traces: int) -> Dict[str, int]:
        """Get statistics about anomaly distribution for a given number of traces."""
        stats = {
            'normal': int(total_traces * 0.9),
            'anomalous_total': int(total_traces * 0.1)
        }
        
        # Break down anomalous traces by severity
        for factor, percentage in self.ANOMALY_DISTRIBUTION.items():
            key = f'anomalous_{factor}x'
            stats[key] = int(total_traces * percentage)
        
        return stats
