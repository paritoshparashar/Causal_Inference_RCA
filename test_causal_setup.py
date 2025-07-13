#!/usr/bin/env python3
"""
Test Causal Inference Setup

This script tests whether the causal inference modules are properly set up
and can process the trace data.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✓ pandas imported")
    except ImportError as e:
        print(f"✗ pandas failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported")
    except ImportError as e:
        print(f"✗ numpy failed: {e}")
        return False
    
    try:
        import networkx as nx
        print("✓ networkx imported")
    except ImportError as e:
        print(f"✗ networkx failed: {e}")
        return False
    
    try:
        from dowhy import gcm
        print("✓ DoWhy/GCM imported")
        dowhy_available = True
    except ImportError as e:
        print(f"✗ DoWhy failed: {e}")
        dowhy_available = False
    
    try:
        from src.causal_inference.data_preparation import LatencyDataProcessor
        print("✓ LatencyDataProcessor imported")
    except ImportError as e:
        print(f"✗ LatencyDataProcessor failed: {e}")
        return False
    
    try:
        from src.causal_inference.causal_model import CausalGraphBuilder
        print("✓ CausalGraphBuilder imported")
    except ImportError as e:
        print(f"✗ CausalGraphBuilder failed: {e}")
        return False
    
    try:
        from src.causal_inference.root_cause_analysis import RootCauseAnalyzer
        print("✓ RootCauseAnalyzer imported")
    except ImportError as e:
        print(f"✗ RootCauseAnalyzer failed: {e}")
        return False
    
    return dowhy_available

def test_data_processing():
    """Test data processing functionality."""
    print("\nTesting data processing...")
    
    # Check if required files exist
    trace_file = Path("output/traces/collected_traces_20250713_230056.json")
    dependency_file = Path("output/analysis/dependency_analysis_20250713_230114.json")
    
    if not trace_file.exists():
        print(f"✗ Trace file not found: {trace_file}")
        return False
    
    if not dependency_file.exists():
        print(f"✗ Dependency file not found: {dependency_file}")
        return False
    
    print("✓ Required files found")
    
    try:
        from src.causal_inference.data_preparation import LatencyDataProcessor
        
        processor = LatencyDataProcessor()
        processor.load_trace_data(str(trace_file))
        processor.load_dependency_graph(str(dependency_file))
        
        latencies = processor.extract_service_latencies()
        normal_data = processor.get_normal_data()
        anomalous_data = processor.get_anomalous_data()
        
        print(f"✓ Extracted latencies for {len(latencies.columns)} services")
        print(f"✓ Normal data: {len(normal_data)} samples")
        print(f"✓ Anomalous data: {len(anomalous_data)} samples")
        
        return True
        
    except Exception as e:
        print(f"✗ Data processing failed: {e}")
        return False

def test_causal_graph():
    """Test causal graph construction."""
    print("\nTesting causal graph construction...")
    
    try:
        from src.causal_inference.causal_model import CausalGraphBuilder
        
        dependency_file = Path("output/analysis/dependency_analysis_20250713_230114.json")
        
        builder = CausalGraphBuilder()
        builder.load_dependencies(str(dependency_file))
        graph = builder.build_causal_graph()
        
        print(f"✓ Causal graph built with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        print(f"  Root nodes: {builder.get_root_nodes()}")
        print(f"  Leaf nodes: {builder.get_leaf_nodes()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Causal graph construction failed: {e}")
        return False

def main():
    print("Causal Inference Setup Test")
    print("=" * 40)
    
    # Test imports
    dowhy_available = test_imports()
    
    # Test data processing
    data_ok = test_data_processing()
    
    # Test causal graph
    graph_ok = test_causal_graph()
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print(f"✓ Basic imports: {'OK' if data_ok and graph_ok else 'FAILED'}")
    print(f"{'✓' if dowhy_available else '✗'} DoWhy/GCM: {'OK' if dowhy_available else 'MISSING (install with: pip install dowhy)'}")
    print(f"✓ Data processing: {'OK' if data_ok else 'FAILED'}")
    print(f"✓ Causal graph: {'OK' if graph_ok else 'FAILED'}")
    
    if dowhy_available and data_ok and graph_ok:
        print("\n🎉 All tests passed! Causal inference is ready to use.")
        return 0
    elif data_ok and graph_ok:
        print("\n⚠️  Basic functionality works, but install DoWhy for full causal inference:")
        print("   pip install dowhy")
        return 0
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
