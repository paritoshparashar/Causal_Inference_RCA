#!/usr/bin/env python3
"""
Main Pipeline for Trace Collection and Dependency Analysis

This script orchestrates the complete pipeline:
1. Collects traces from Jaeger using JaegerClient
2. Detects anomalous traces using AnomalyDetector
3. Analyzes service dependencies using DependencyGraphBuilder
4. Generates various output formats (JSON, DOT, adjacency list)

Usage:
    python main.py                           # Single run with Jaeger collection
    python main.py --continuous             # Continuous mode
    python main.py --analyze-only <file>    # Only analyze existing traces
"""

import sys
import os
import time
import logging
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trace_collection import Config, JaegerClient, AnomalyDetector
from dependency_analysis import DependencyGraphBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main-pipeline")


def collect_traces(config):
    """
    Step 1: Collect traces from Jaeger and detect anomalies.
    Returns: (traces_file_path, anomalous_traces_file_path) or (None, None) if no traces
    """
    logger.info("=== STEP 1: TRACE COLLECTION ===")
    
    # Initialize Jaeger client and anomaly detector
    jaeger_client = JaegerClient(config)
    anomaly_detector = AnomalyDetector(config.LATENCY_PERCENTILE)
    
    # Calculate time range for trace collection
    end_time = int(time.time() * 1000000)  # microseconds
    start_time = end_time - (config.PAST_INTERVAL * 1000000)
    
    logger.info(f"Collecting traces from {datetime.fromtimestamp(start_time/1000000)} to {datetime.fromtimestamp(end_time/1000000)}")
    
    # Fetch traces from Jaeger
    traces = jaeger_client.fetch_traces(start_time, end_time)
    
    if not traces:
        logger.warning("No traces collected from Jaeger")
        return None, None
    
    logger.info(f"Successfully collected {len(traces)} traces")
    
    # Detect anomalous traces
    logger.info("Detecting anomalous traces...")
    anomalous_traces = anomaly_detector.detect_anomalies(traces)
    logger.info(f"Found {len(anomalous_traces)} anomalous traces")
    
    # Save traces to files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output/traces")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all collected traces
    traces_file = output_dir / f"collected_traces_{timestamp}.json"
    with open(traces_file, 'w') as f:
        json.dump(traces, f, indent=2)
    logger.info(f"Saved {len(traces)} traces to: {traces_file}")
    
    # Save anomalous traces separately
    anomalous_file = output_dir / f"anomalous_traces_{timestamp}.json"
    with open(anomalous_file, 'w') as f:
        json.dump(anomalous_traces, f, indent=2)
    logger.info(f"Saved {len(anomalous_traces)} anomalous traces to: {anomalous_file}")
    
    return str(traces_file), str(anomalous_file)


def analyze_dependencies(traces_file, timestamp=None):
    """
    Step 2: Analyze service dependencies from traces and generate outputs.
    Returns: list of generated output file paths
    """
    logger.info("=== STEP 2: DEPENDENCY ANALYSIS ===")
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Analyzing dependencies from: {traces_file}")
    
    # Load traces from file
    try:
        with open(traces_file, 'r') as f:
            data = json.load(f)
            
        # Handle different trace file formats
        if isinstance(data, dict) and 'data' in data:
            traces = data['data']
        elif isinstance(data, list):
            traces = data
        else:
            traces = [data]
            
    except Exception as e:
        logger.error(f"Failed to load traces from {traces_file}: {e}")
        raise
    
    logger.info(f"Loaded {len(traces)} traces for dependency analysis")
    
    # Build dependency graph
    builder = DependencyGraphBuilder()
    builder.add_traces(traces)
    
    # Create output directory
    output_dir = Path("output/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = []
    
    # Generate JSON summary
    logger.info("Generating JSON analysis...")
    json_file = output_dir / f"dependency_analysis_{timestamp}.json"
    summary = builder.get_dependency_summary()
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    output_files.append(str(json_file))
    logger.info(f"✓ JSON analysis saved: {json_file}")
    
    # Generate NetworkX graph for visualization
    graph = builder.build_networkx_graph()
    
    # Generate DOT file for Graphviz
    logger.info("Generating DOT file...")
    dot_file = output_dir / f"dependency_graph_{timestamp}.dot"
    dot_content = generate_dot_graph(graph)
    with open(dot_file, 'w') as f:
        f.write(dot_content)
    output_files.append(str(dot_file))
    logger.info(f"✓ DOT file saved: {dot_file}")
    
    # Generate human-readable adjacency list
    logger.info("Generating adjacency list...")
    adj_file = output_dir / f"dependency_graph_{timestamp}_adjacency.txt"
    generate_adjacency_list(graph, adj_file)
    output_files.append(str(adj_file))
    logger.info(f"✓ Adjacency list saved: {adj_file}")
    
    # Try to generate PNG visualization if Graphviz is available
    try:
        import subprocess
        png_file = output_dir / f"dependency_graph_{timestamp}.png"
        result = subprocess.run(['dot', '-Tpng', str(dot_file), '-o', str(png_file)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            output_files.append(str(png_file))
            logger.info(f"✓ PNG visualization saved: {png_file}")
        else:
            logger.warning("Graphviz available but PNG generation failed")
    except FileNotFoundError:
        logger.warning("Graphviz not installed - PNG visualization skipped")
        logger.info("  To install: brew install graphviz (macOS) or apt-get install graphviz (Ubuntu)")
    
    # Print summary
    logger.info("Dependency analysis completed!")
    logger.info(f"Found {len(graph.nodes())} services with {len(graph.edges())} dependencies")
    logger.info(f"Generated {len(output_files)} output files")
    
    return output_files


def generate_dot_graph(graph):
    """Generate DOT format string for Graphviz visualization."""
    dot_lines = ["digraph service_dependencies {"]
    dot_lines.append("  rankdir=LR;")
    dot_lines.append("  node [shape=box, style=filled, fillcolor=lightblue];")
    dot_lines.append("  edge [color=gray];")
    dot_lines.append("")
    
    # Add nodes (services)
    for node in graph.nodes():
        dot_lines.append(f'  "{node}";')
    
    dot_lines.append("")
    
    # Add edges (dependencies) with call counts
    for u, v in graph.edges():
        weight = graph[u][v].get('weight', 1)
        dot_lines.append(f'  "{u}" -> "{v}" [label="{weight}"];')
    
    dot_lines.append("}")
    return "\n".join(dot_lines)


def generate_adjacency_list(graph, output_file):
    """Generate human-readable adjacency list file."""
    with open(output_file, 'w') as f:
        f.write("SERVICE DEPENDENCY ADJACENCY LIST\n")
        f.write("=" * 50 + "\n\n")
        
        for service in sorted(graph.nodes()):
            calls = list(graph.successors(service))  # Services this service calls
            called_by = list(graph.predecessors(service))  # Services that call this service
            
            f.write(f"Service: {service}\n")
            f.write(f"  Calls: {calls if calls else ['(none)']}\n")
            f.write(f"  Called by: {called_by if called_by else ['(none)']}\n")
            f.write("\n")


def run_complete_pipeline(config):
    """Run the complete pipeline: trace collection + dependency analysis."""
    try:
        # Step 1: Collect traces from Jaeger
        traces_file, anomalous_file = collect_traces(config)
        
        if traces_file is None:
            logger.warning("No traces collected - pipeline stopped")
            return
        
        # Step 2: Analyze dependencies
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = analyze_dependencies(traces_file, timestamp)
        
        # Pipeline completion summary
        logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
        logger.info("Generated files:")
        logger.info(f"  📊 Traces: {traces_file}")
        logger.info(f"  🚨 Anomalies: {anomalous_file}")
        for output_file in output_files:
            logger.info(f"  📈 Analysis: {output_file}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Trace Collection and Dependency Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py                              # Single pipeline run
  python main.py --continuous --interval 600  # Continuous mode (10 min intervals)
  python main.py --analyze-only traces.json   # Only analyze existing traces
        '''
    )
    
    parser.add_argument('--continuous', action='store_true',
                       help='Run in continuous mode')
    parser.add_argument('--interval', type=int, default=300,
                       help='Interval between collections in continuous mode (seconds, default: 300)')
    parser.add_argument('--analyze-only', type=str, metavar='TRACES_FILE',
                       help='Skip trace collection and only analyze existing traces file')
    
    args = parser.parse_args()
    
    # Analyze-only mode (no trace collection)
    if args.analyze_only:
        if not os.path.exists(args.analyze_only):
            logger.error(f"Traces file not found: {args.analyze_only}")
            return 1
            
        try:
            output_files = analyze_dependencies(args.analyze_only)
            logger.info("✅ Analysis completed successfully!")
            return 0
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return 1
    
    # Full pipeline mode (trace collection + analysis)
    config = Config()
    logger.info("🚀 Starting Trace Collection and Dependency Analysis Pipeline")
    logger.info(f"📡 Jaeger endpoint: {config.jaeger_endpoint}")
    logger.info(f"🎯 Target service: {config.SERVICE_NAME}")
    
    if args.continuous:
        logger.info(f"🔄 Continuous mode enabled (interval: {args.interval}s)")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                run_complete_pipeline(config)
                logger.info(f"⏳ Waiting {args.interval} seconds until next collection...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("🛑 Pipeline stopped by user")
    else:
        logger.info("🎯 Single run mode")
        run_complete_pipeline(config)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())