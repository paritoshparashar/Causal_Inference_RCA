#!/usr/bin/env python3
"""
Simple graph visualization tool - prints a nice ASCII representation
"""

import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.dependency_analysis.dependency_graph import DependencyGraphBuilder

def print_graph_table(graph):
    """Print dependency graph in a nice table format."""
    print("\n" + "="*80)
    print("SERVICE DEPENDENCY DETAILS")
    print("="*80)
    
    # Get all edges with weights
    edges_with_weights = []
    for u, v in graph.edges():
        weight = graph[u][v].get('weight', 1)
        edges_with_weights.append((u, v, weight))
    
    # Sort by weight (descending)
    edges_with_weights.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'Caller Service':<15} {'→':<3} {'Called Service':<15} {'Call Count':<12} {'Visual':<20}")
    print("-" * 80)
    
    max_weight = max(edge[2] for edge in edges_with_weights) if edges_with_weights else 1
    
    for caller, callee, weight in edges_with_weights:
        # Create visual bar representation
        bar_length = int((weight / max_weight) * 15)
        visual_bar = "█" * bar_length + "░" * (15 - bar_length)
        
        print(f"{caller:<15} → {callee:<15} {weight:>8,} {visual_bar}")
    
    print("-" * 80)
    print(f"Total dependencies: {len(edges_with_weights)}")
    print(f"Total calls: {sum(edge[2] for edge in edges_with_weights):,}")

def print_service_summary(graph):
    """Print a summary of each service."""
    print("\n" + "="*60)
    print("SERVICE SUMMARY")
    print("="*60)
    
    services = list(graph.nodes())
    services.sort()
    
    for service in services:
        incoming = list(graph.predecessors(service))
        outgoing = list(graph.successors(service))
        
        incoming_calls = sum(graph[pred][service].get('weight', 1) for pred in incoming)
        outgoing_calls = sum(graph[service][succ].get('weight', 1) for succ in outgoing)
        
        print(f"\n📦 {service.upper()}")
        print(f"   Receives calls from: {incoming if incoming else ['(none - root service)']}")
        print(f"   Calls: {outgoing if outgoing else ['(none - leaf service)']}")
        print(f"   Incoming call volume: {incoming_calls:,}")
        print(f"   Outgoing call volume: {outgoing_calls:,}")
        
        if not incoming:
            print("   🔴 ROOT SERVICE (entry point)")
        if not outgoing:
            print("   🟢 LEAF SERVICE (no downstream calls)")

def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_graph.py <trace_file.json>")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    
    try:
        # Load and analyze traces
        with open(trace_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'data' in data:
            traces = data['data']
        elif isinstance(data, list):
            traces = data
        else:
            traces = [data]
        
        print(f"📊 Analyzing {len(traces)} traces from {trace_file}")
        
        # Build graph
        builder = DependencyGraphBuilder()
        builder.add_traces(traces)
        graph = builder.build_networkx_graph()
        
        # Print visualizations
        print_graph_table(graph)
        print_service_summary(graph)
        
        print("\n" + "="*60)
        print("For graphical visualization, run:")
        print(f"python analyze_dependencies.py {trace_file} --visualize")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
