"""
Service Topology Builder

Defines the 10-service microservice architecture with non-linear dependencies.
"""

from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class ServiceConfig:
    """Configuration for a single service."""
    name: str
    baseline_latency_ms: float
    operation_name: str
    downstream_services: List[str]


class ServiceTopology:
    """
    Manages the 12-service microservice topology with realistic e-commerce dependencies.
    
    Architecture (More Complex - Realistic E-commerce Flow):
                        frontend (100ms)
                           ↓
                    api_gateway (50ms)
              ↙        ↓         ↓        ↘
      auth_svc    order_svc   search_svc  user_profile_svc
       (30ms)      (40ms)      (35ms)       (25ms)
         ↓           ↓           ↓            ↓
      user_db    payment_svc   product_db   user_db (shared)
       (20ms)      (25ms)      (15ms)      (shared)
         ↓           ↓           ↓
    session_cache inventory_svc ←┘ (stock validation)
       (5ms)        (30ms)
                      ↓
                 shipping_svc (20ms)
                 ↙         ↘
         tracking_svc   notification_svc
           (15ms)          (10ms)
    """
    
    def __init__(self):
        self.services = self._build_service_configs()
        self.dependency_graph = self._build_dependency_graph()
        
    def _build_service_configs(self) -> Dict[str, ServiceConfig]:
        """Build the service configurations with baseline latencies."""
        return {
            'frontend': ServiceConfig(
                name='frontend',
                baseline_latency_ms=100.0,
                operation_name='frontend_request',
                downstream_services=['api_gateway']
            ),
            'api_gateway': ServiceConfig(
                name='api_gateway',
                baseline_latency_ms=50.0,
                operation_name='gateway_process',
                downstream_services=['auth_svc', 'order_svc', 'search_svc', 'user_profile_svc']
            ),
            'auth_svc': ServiceConfig(
                name='auth_svc',
                baseline_latency_ms=30.0,
                operation_name='authenticate_user',
                downstream_services=['user_db', 'session_cache']
            ),
            'order_svc': ServiceConfig(
                name='order_svc',
                baseline_latency_ms=40.0,
                operation_name='process_order',
                downstream_services=['payment_svc']
            ),
            'search_svc': ServiceConfig(
                name='search_svc',
                baseline_latency_ms=35.0,
                operation_name='search_products',
                downstream_services=['product_db', 'inventory_svc']  # Also checks stock
            ),
            'user_profile_svc': ServiceConfig(
                name='user_profile_svc',
                baseline_latency_ms=25.0,
                operation_name='get_user_profile',
                downstream_services=['user_db']  # Shared with auth_svc
            ),
            'payment_svc': ServiceConfig(
                name='payment_svc',
                baseline_latency_ms=25.0,
                operation_name='process_payment',
                downstream_services=['inventory_svc']  # Reserve inventory
            ),
            'inventory_svc': ServiceConfig(
                name='inventory_svc',
                baseline_latency_ms=30.0,
                operation_name='check_inventory',
                downstream_services=['shipping_svc']  # Convergence point
            ),
            'shipping_svc': ServiceConfig(
                name='shipping_svc',
                baseline_latency_ms=20.0,
                operation_name='arrange_shipping',
                downstream_services=['tracking_svc', 'notification_svc']  # Fan-out
            ),
            'tracking_svc': ServiceConfig(
                name='tracking_svc',
                baseline_latency_ms=15.0,
                operation_name='create_tracking',
                downstream_services=[]  # Leaf node
            ),
            'notification_svc': ServiceConfig(
                name='notification_svc',
                baseline_latency_ms=10.0,
                operation_name='send_notification',
                downstream_services=[]  # Leaf node
            ),
            'user_db': ServiceConfig(
                name='user_db',
                baseline_latency_ms=20.0,
                operation_name='query_user_data',
                downstream_services=[]  # Shared resource - leaf node
            ),
            'product_db': ServiceConfig(
                name='product_db',
                baseline_latency_ms=15.0,
                operation_name='query_product_data',
                downstream_services=[]  # Leaf node
            ),
            'session_cache': ServiceConfig(
                name='session_cache',
                baseline_latency_ms=5.0,
                operation_name='manage_session',
                downstream_services=[]  # Fast cache - leaf node
            )
        }
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build the dependency graph from service configs."""
        graph = {}
        for service_name, config in self.services.items():
            graph[service_name] = config.downstream_services
        return graph
    
    def get_service_names(self) -> List[str]:
        """Get all service names in the topology."""
        return list(self.services.keys())
    
    def get_service_config(self, service_name: str) -> ServiceConfig:
        """Get configuration for a specific service."""
        if service_name not in self.services:
            raise ValueError(f"Service '{service_name}' not found in topology")
        return self.services[service_name]
    
    def get_downstream_services(self, service_name: str) -> List[str]:
        """Get direct downstream dependencies for a service."""
        return self.dependency_graph.get(service_name, [])
    
    def get_upstream_services(self, service_name: str) -> List[str]:
        """Get all services that depend on this service (directly or indirectly)."""
        upstream = []
        for service, dependencies in self.dependency_graph.items():
            if service_name in dependencies:
                upstream.append(service)
                # Recursively get upstream services
                upstream.extend(self.get_upstream_services(service))
        return list(set(upstream))  # Remove duplicates
    
    def get_call_chain(self, start_service: str = 'frontend') -> List[str]:
        """
        Get the complete call chain starting from a service.
        Returns services in the order they would be called.
        """
        visited = set()
        call_chain = []
        
        def _traverse(service: str):
            if service in visited:
                return
            visited.add(service)
            call_chain.append(service)
            
            # Add downstream services
            for downstream in self.get_downstream_services(service):
                _traverse(downstream)
        
        _traverse(start_service)
        return call_chain
    
    def is_leaf_service(self, service_name: str) -> bool:
        """Check if a service is a leaf node (no downstream dependencies)."""
        return len(self.get_downstream_services(service_name)) == 0
    
    def validate_service_name(self, service_name: str) -> bool:
        """Validate that a service name exists in the topology."""
        return service_name in self.services
    
    def get_topology_summary(self) -> str:
        """Get a text summary of the topology structure."""
        summary = ["Service Topology (12 services - Complex E-commerce Flow):"]
        summary.append("")
        summary.append("Key Features:")
        summary.append("• Shared Resources: user_db used by auth_svc & user_profile_svc")
        summary.append("• Convergence Points: inventory_svc called by payment_svc & search_svc")
        summary.append("• Fan-out Points: api_gateway (4 services), shipping_svc (2 services)")
        summary.append("• Business Logic: Realistic e-commerce flow with cross-service dependencies")
        summary.append("")
        
        def _add_service_info(service: str, indent: int = 0):
            config = self.get_service_config(service)
            prefix = "  " * indent
            summary.append(f"{prefix}{service} ({config.baseline_latency_ms}ms)")
            
            downstream = self.get_downstream_services(service)
            if downstream:
                for child in downstream:
                    _add_service_info(child, indent + 1)
        
        _add_service_info('frontend')
        return "\n".join(summary)
