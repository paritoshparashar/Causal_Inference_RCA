"""
Dependency Graph Builder for OpenTelemetry Traces

This module builds service dependency graphs from distributed traces,
analyzing parent-child relationships between spans to understand
how services interact with each other.
"""

import networkx as nx
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional

logger = logging.getLogger("dependency-graph")


class DependencyGraphBuilder:
    """Builds service dependency graphs from OpenTelemetry traces."""
    
    def __init__(self):
        self.service_dependencies = defaultdict(set)  # service -> set of dependencies
        self.dependency_counts = defaultdict(int)     # (caller, callee) -> count
        self.service_operations = defaultdict(set)    # service -> set of operations
        self.trace_count = 0
        
    def extract_service_name(self, span: Dict, processes: Dict) -> str:
        """Extract service name from span without any transformations."""
        process_id = span.get('processID', '')
        if process_id in processes:
            service_name = processes[process_id].get('serviceName', 'unknown')
            return service_name
        return 'unknown'
    
    def build_span_tree(self, spans: List[Dict]) -> Dict[str, Dict]:
        """Build a tree structure from spans using parent-child relationships."""
        span_map = {}
        
        # First pass: create span lookup
        for span in spans:
            span_id = span.get('spanID')
            if span_id:
                span_map[span_id] = {
                    'span': span,
                    'children': []
                }
        
        # Second pass: establish parent-child relationships
        root_spans = []
        for span in spans:
            span_id = span.get('spanID')
            
            # Look for parent references
            parent_span_id = None
            references = span.get('references', [])
            
            for ref in references:
                if ref.get('refType') == 'CHILD_OF':
                    parent_span_id = ref.get('spanID')
                    break
            
            if parent_span_id and parent_span_id in span_map:
                # Add to parent's children
                span_map[parent_span_id]['children'].append(span_id)
            else:
                # Root span (no parent)
                root_spans.append(span_id)
        
        return span_map, root_spans
    
    def extract_dependencies_from_trace(self, trace: Dict) -> List[Tuple[str, str]]:
        """Extract service dependencies from a single trace."""
        if 'spans' not in trace:
            return []
        
        spans = trace['spans']
        processes = trace.get('processes', {})
        dependencies = []
        
        # Build span tree
        span_map, root_spans = self.build_span_tree(spans)
        
        # Extract service for each span
        span_services = {}
        for span in spans:
            span_id = span.get('spanID')
            service_name = self.extract_service_name(span, processes)
            operation_name = span.get('operationName', 'unknown')
            
            span_services[span_id] = service_name
            self.service_operations[service_name].add(operation_name)
        
        # Find cross-service dependencies
        for span in spans:
            span_id = span.get('spanID')
            current_service = span_services.get(span_id)
            
            if not current_service or span_id not in span_map:
                continue
            
            # Check children for cross-service calls
            for child_span_id in span_map[span_id]['children']:
                child_service = span_services.get(child_span_id)
                
                if child_service and current_service != child_service:
                    # Found a cross-service dependency: current_service -> child_service
                    dependency = (current_service, child_service)
                    dependencies.append(dependency)
                    
                    logger.debug(f"Found dependency: {current_service} -> {child_service}")
        
        return dependencies
    
    def add_trace(self, trace: Dict) -> None:
        """Add a single trace to the dependency analysis."""
        self.trace_count += 1
        
        dependencies = self.extract_dependencies_from_trace(trace)
        
        for caller, callee in dependencies:
            self.service_dependencies[caller].add(callee)
            self.dependency_counts[(caller, callee)] += 1
    
    def add_traces(self, traces: List[Dict]) -> None:
        """Add multiple traces to the dependency analysis."""
        logger.info(f"Processing {len(traces)} traces for dependency extraction...")
        
        for i, trace in enumerate(traces):
            try:
                # Handle different trace formats
                if isinstance(trace, dict):
                    if 'data' in trace:
                        # Nested format
                        for trace_item in trace['data']:
                            self.add_trace(trace_item)
                    else:
                        # Direct format
                        self.add_trace(trace)
                        
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1} traces...")
                    
            except Exception as e:
                logger.warning(f"Error processing trace {i}: {e}")
        
        logger.info(f"Completed processing {self.trace_count} traces")
        logger.info(f"Found {len(self.service_dependencies)} services")
        logger.info(f"Found {sum(len(deps) for deps in self.service_dependencies.values())} unique dependencies")
    
    def build_networkx_graph(self) -> nx.DiGraph:
        """Build a NetworkX directed graph from the dependencies."""
        graph = nx.DiGraph()
        
        # Add all services as nodes
        all_services = set()
        for caller, callees in self.service_dependencies.items():
            all_services.add(caller)
            all_services.update(callees)
        
        for service in all_services:
            graph.add_node(service)
        
        # Add edges with weights (call counts)
        for (caller, callee), count in self.dependency_counts.items():
            graph.add_edge(caller, callee, weight=count)
        
        return graph
    
    def get_dependency_summary(self) -> Dict:
        """Get a summary of discovered dependencies."""
        total_dependencies = sum(len(deps) for deps in self.service_dependencies.values())
        
        # Find most frequently called services
        callee_counts = Counter()
        for deps in self.service_dependencies.values():
            for callee in deps:
                callee_counts[callee] += 1
        
        # Find services with most outgoing calls
        caller_counts = {service: len(deps) for service, deps in self.service_dependencies.items()}
        
        return {
            'total_traces_processed': self.trace_count,
            'total_services': len(set(list(self.service_dependencies.keys()) + 
                                   [callee for deps in self.service_dependencies.values() for callee in deps])),
            'total_dependencies': total_dependencies,
            'services_with_dependencies': list(self.service_dependencies.keys()),
            'most_called_services': callee_counts.most_common(5),
            'services_with_most_calls': sorted(caller_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'dependency_details': {service: list(deps) for service, deps in self.service_dependencies.items()}
        }
    
    def print_summary(self) -> None:
        """Print a human-readable summary of the dependency graph."""
        summary = self.get_dependency_summary()
        
        print("=" * 60)
        print("SERVICE DEPENDENCY GRAPH SUMMARY")
        print("=" * 60)
        print(f"Traces processed: {summary['total_traces_processed']}")
        print(f"Services discovered: {summary['total_services']}")
        print(f"Dependencies found: {summary['total_dependencies']}")
        print()
        
        print("SERVICES WITH OUTGOING DEPENDENCIES:")
        for service, deps in summary['dependency_details'].items():
            print(f"  {service} -> {list(deps)}")
        print()
        
        print("MOST FREQUENTLY CALLED SERVICES:")
        for service, count in summary['most_called_services']:
            print(f"  {service}: called by {count} different services")
        print()
        
        print("SERVICES MAKING THE MOST CALLS:")
        for service, count in summary['services_with_most_calls']:
            print(f"  {service}: calls {count} other services")
        
        print("=" * 60)


def analyze_trace_dependencies(traces: List[Dict]) -> Tuple[nx.DiGraph, Dict]:
    """
    Analyze traces and return dependency graph and summary.
    
    Args:
        traces: List of OpenTelemetry traces
        
    Returns:
        Tuple of (NetworkX graph, summary dictionary)
    """
    builder = DependencyGraphBuilder()
    builder.add_traces(traces)
    
    graph = builder.build_networkx_graph()
    summary = builder.get_dependency_summary()
    
    return graph, summary


# Example usage and testing
if __name__ == "__main__":
    import json
    
    # Test with sample data
    try:
        with open('../../traces.json', 'r') as f:
            sample_traces = json.load(f)
        
        print("Testing dependency extraction with sample traces...")
        graph, summary = analyze_trace_dependencies(sample_traces)
        
        builder = DependencyGraphBuilder()
        builder.add_traces(sample_traces)
        builder.print_summary()
        
        print(f"\nNetworkX Graph Summary:")
        print(f"Nodes: {list(graph.nodes())}")
        print(f"Edges: {list(graph.edges())}")
        
    except FileNotFoundError:
        print("No sample traces file found. Create some test data to verify the implementation.")
