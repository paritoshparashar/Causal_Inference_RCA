#!/usr/bin/env python3
"""
Dependency Graph Analysis Tool

This script analyzes OpenTelemetry traces to build and visualize
service dependency graphs.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.dependency_analysis.dependency_graph import DependencyGraphBuilder, analyze_trace_dependencies

def find_trace_file(filename_or_path: str = None):
    """Find trace file in output/traces directory or use latest if not specified."""
    # Base directory for trace files
    traces_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'traces')
    traces_dir = os.path.abspath(traces_dir)
    
    if filename_or_path is None:
        # Find the most recent trace file
        if not os.path.exists(traces_dir):
            raise FileNotFoundError(f"Traces directory not found: {traces_dir}")
        
        trace_files = [f for f in os.listdir(traces_dir) 
                      if f.startswith('collected_traces_') and f.endswith('.json')]
        
        if not trace_files:
            raise FileNotFoundError(f"No trace files found in {traces_dir}")
        
        # Sort by filename (which includes timestamp) to get the latest
        trace_files.sort(reverse=True)
        latest_file = trace_files[0]
        full_path = os.path.join(traces_dir, latest_file)
        print(f"Using latest trace file: {latest_file}")
        
    elif os.path.isabs(filename_or_path):
        # Absolute path provided
        full_path = filename_or_path
        
    elif os.path.sep in filename_or_path or filename_or_path.startswith('.'):
        # Relative path provided - resolve from current directory
        full_path = os.path.abspath(filename_or_path)
        
    else:
        # Just filename provided - look in traces directory
        full_path = os.path.join(traces_dir, filename_or_path)
    
    if not os.path.exists(full_path):
        # If file not found, suggest available files
        if os.path.exists(traces_dir):
            available_files = [f for f in os.listdir(traces_dir) 
                             if f.endswith('.json')]
            if available_files:
                print(f"Available trace files in {traces_dir}:")
                for f in sorted(available_files, reverse=True):
                    print(f"  {f}")
        raise FileNotFoundError(f"Trace file not found: {full_path}")
    
    return full_path

def load_traces_from_file(file_path: str):
    """Load traces from a JSON file."""
    print(f"Loading traces from {file_path}...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle different file formats
    if isinstance(data, dict):
        if 'data' in data:
            traces = data['data']
        else:
            traces = [data]
    elif isinstance(data, list):
        traces = data
    else:
        traces = [data]
    
    print(f"Loaded {len(traces)} traces")
    return traces

def create_hierarchical_layout(graph):
    """Create a hierarchical layout for better service dependency visualization."""
    try:
        import networkx as nx
        import math
        
        pos = {}
        
        # Identify service types based on naming patterns and graph structure
        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        
        # Layer 1 (Top): Root services (traffic generators)
        root_services = roots if roots else []
        
        # Layer 2 (Middle): Orchestrator services (high out-degree, like frontend)
        orchestrators = [n for n in graph.nodes() 
                        if graph.out_degree(n) >= 3 and n not in root_services]
        
        # Layer 3 (Middle-Lower): Intermediate services (both in and out connections)
        intermediates = [n for n in graph.nodes() 
                        if graph.in_degree(n) > 0 and graph.out_degree(n) > 0 
                        and n not in root_services and n not in orchestrators]
        
        # Layer 4 (Bottom): Leaf services (backend services)
        leaf_services = [n for n in leaves if n not in root_services]
        
        # Arrange nodes in layers
        layers = [
            ("Root Services", root_services, 0.9),      # Top layer
            ("Orchestrators", orchestrators, 0.6),      # Upper middle
            ("Intermediates", intermediates, 0.3),      # Lower middle  
            ("Backend Services", leaf_services, 0.0)    # Bottom layer
        ]
        
        # Position nodes in each layer
        for layer_name, services, y_pos in layers:
            if not services:
                continue
                
            # Spread services horizontally within each layer
            num_services = len(services)
            if num_services == 1:
                x_positions = [0.0]
            else:
                # Spread evenly with good spacing
                spacing = min(1.6 / max(1, num_services - 1), 0.4)
                start_x = -spacing * (num_services - 1) / 2
                x_positions = [start_x + i * spacing for i in range(num_services)]
            
            for i, service in enumerate(services):
                pos[service] = (x_positions[i], y_pos)
        
        # Ensure all nodes are positioned
        unpositioned = set(graph.nodes()) - set(pos.keys())
        if unpositioned:
            # Position remaining nodes in middle layer
            num_remaining = len(unpositioned)
            for i, service in enumerate(unpositioned):
                x = -0.5 + (i * 1.0 / max(1, num_remaining - 1))
                pos[service] = (x, 0.45)  # Middle layer
        
        return pos
        
    except Exception as e:
        print(f"Hierarchical layout failed: {e}, falling back to spring layout")
        return None

def visualize_graph(graph, output_file=None):
    """Visualize the dependency graph using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        import math
        
        plt.figure(figsize=(16, 12))
        
        # Create hierarchical layout based on service types
        pos = create_hierarchical_layout(graph)
        
        # Fallback to improved spring layout if hierarchical fails
        if not pos:
            pos = nx.spring_layout(graph, k=8, iterations=200, seed=42)
        
        # Draw nodes with better styling and larger size for better visibility
        nx.draw_networkx_nodes(graph, pos, node_color='lightsteelblue', 
                              node_size=5000, alpha=0.9, linewidths=2, edgecolors='navy')
        
        # Draw labels with better formatting
        nx.draw_networkx_labels(graph, pos, font_size=11, font_weight='bold', font_color='darkblue')
        
        # Draw edges with improved weight scaling
        edges = graph.edges()
        weights = [graph[u][v].get('weight', 1) for u, v in edges]
        
        # Improved edge thickness scaling with logarithmic approach
        if weights:
            max_weight = max(weights)
            min_weight = min(weights)
            
            # Use logarithmic scaling for better visibility of all edges
            import math
            
            # Set absolute minimum thickness for visibility
            min_thickness = 2.5  # Increased minimum thickness
            max_thickness = 8.0  # Increased maximum thickness
            
            if max_weight == min_weight:
                normalized_weights = [4.0] * len(weights)
            else:
                # Logarithmic scaling: log(weight + 1) to handle edge case
                log_weights = [math.log(w + 1) for w in weights]
                log_min = min(log_weights)
                log_max = max(log_weights)
                
                if log_max == log_min:
                    normalized_weights = [4.0] * len(weights)
                else:
                    normalized_weights = [
                        min_thickness + (log_w - log_min) / (log_max - log_min) * (max_thickness - min_thickness)
                        for log_w in log_weights
                    ]
        else:
            normalized_weights = [3.0]  # Default thickness
        
        # Draw edges with better visibility and curved connections
        nx.draw_networkx_edges(graph, pos, width=normalized_weights,
                              edge_color='darkblue', arrows=True, 
                              arrowsize=25, arrowstyle='->', alpha=0.8,
                              connectionstyle='arc3,rad=0.1')  # Curved edges
        
        # Add edge labels to show actual weights with better positioning
        edge_labels = {(u, v): f"{graph[u][v].get('weight', 1):,}" for u, v in edges}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=9, 
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        plt.title("Service Dependency Graph\n(Hierarchical Layout - Edge thickness shows call frequency)", size=18, pad=30)
        
        # Add improved legend with more information
        if weights:
            legend_text = f"Call Frequencies:\nMin: {min(weights):,}\nMax: {max(weights):,}\nTotal: {sum(weights):,}\n\nScaling: Logarithmic\nLayout: Hierarchical"
            plt.text(0.02, 0.98, legend_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                    fontsize=11)
        
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to {output_file}")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error creating visualization: {e}")

def export_graph_formats(graph, base_filename):
    """Export graph in multiple formats."""
    
    # Export as DOT format (GraphViz)
    try:
        import networkx as nx
        dot_file = f"{base_filename}.dot"
        nx.drawing.nx_pydot.write_dot(graph, dot_file)
        print(f"DOT format exported to {dot_file}")
    except:
        print("Could not export DOT format (pydot not available)")
    
    # Export as JSON
    try:
        import networkx as nx
        json_data = nx.node_link_data(graph)
        json_file = f"{base_filename}.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON format exported to {json_file}")
    except Exception as e:
        print(f"Could not export JSON format: {e}")
    
    # Export as adjacency list
    try:
        adj_file = f"{base_filename}_adjacency.txt"
        with open(adj_file, 'w') as f:
            f.write("# Service Dependency Graph - Adjacency List\n")
            f.write("# Format: service -> [dependencies]\n\n")
            for node in graph.nodes():
                successors = list(graph.successors(node))
                f.write(f"{node} -> {successors}\n")
        print(f"Adjacency list exported to {adj_file}")
    except Exception as e:
        print(f"Could not export adjacency list: {e}")

def main():
    parser = argparse.ArgumentParser(description='Analyze service dependencies from OpenTelemetry traces')
    parser.add_argument('trace_file', nargs='?', help='Path to JSON file containing traces, or just filename to look in output/traces/ (uses latest if not specified)')
    parser.add_argument('--output', '-o', help='Base filename for output files')
    parser.add_argument('--visualize', '-v', action='store_true', help='Show graph visualization')
    parser.add_argument('--export', '-e', action='store_true', help='Export graph in multiple formats')
    
    args = parser.parse_args()
    
    try:
        # Find the trace file
        trace_file_path = find_trace_file(args.trace_file)
        
        # Load traces
        traces = load_traces_from_file(trace_file_path)
        
        # Build dependency graph
        print("\nBuilding dependency graph...")
        builder = DependencyGraphBuilder()
        builder.add_traces(traces)
        
        # Print summary
        builder.print_summary()
        
        # Get NetworkX graph
        graph = builder.build_networkx_graph()
        
        print(f"\nNetworkX Graph Details:")
        print(f"Nodes ({len(graph.nodes())}): {list(graph.nodes())}")
        print(f"Edges ({len(graph.edges())}): {list(graph.edges())}")
        
        # Calculate graph metrics
        try:
            import networkx as nx
            
            print(f"\nGraph Metrics:")
            print(f"Number of nodes: {graph.number_of_nodes()}")
            print(f"Number of edges: {graph.number_of_edges()}")
            print(f"Is DAG (Directed Acyclic Graph): {nx.is_directed_acyclic_graph(graph)}")
            
            if graph.number_of_nodes() > 0:
                print(f"Average degree: {sum(dict(graph.degree()).values()) / graph.number_of_nodes():.2f}")
                
                # Find root and leaf services
                roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
                leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]
                
                print(f"Root services (no incoming calls): {roots}")
                print(f"Leaf services (no outgoing calls): {leaves}")
                
        except Exception as e:
            print(f"Error calculating graph metrics: {e}")
        
        # Create output directory for analysis results
        output_base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'analysis')
        output_base_dir = os.path.abspath(output_base_dir)
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Export if requested
        if args.export:
            if args.output:
                base_name = os.path.join(output_base_dir, args.output)
            else:
                # Create filename based on trace file
                trace_basename = Path(trace_file_path).stem
                base_name = os.path.join(output_base_dir, f"{trace_basename}_dependency_graph")
            export_graph_formats(graph, base_name)
        
        # Visualize if requested
        if args.visualize:
            if args.output:
                output_img = os.path.join(output_base_dir, f"{args.output}_graph.png")
            else:
                # Create filename based on trace file
                trace_basename = Path(trace_file_path).stem
                output_img = os.path.join(output_base_dir, f"{trace_basename}_dependency_graph.png")
            visualize_graph(graph, output_img)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
