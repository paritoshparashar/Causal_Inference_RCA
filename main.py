#!/usr/bin/env python3
"""
Main Pipeline for Trace Collection, Dependency Analysis, and Causal Inference

This script orchestrates the complete pipeline:
1. Collects traces from Jaeger using JaegerClient
2. Detects anomalous traces using AnomalyDetector
3. Analyzes service dependencies using DependencyGraphBuilder
4. Generates various output formats (JSON, DOT, adjacency list)
5. Performs root cause analysis using causal inference (optional)

Usage:
    python main.py                           # Single run with Jaeger collection
    python main.py --continuous             # Continuous mode
    python main.py --analyze-only <file>    # Only analyze existing traces
    python main.py --causal-inference       # Include causal inference RCA
    python main.py --rca-target frontend    # Specify target service for RCA
"""

import sys
import os
import time
import logging
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trace_collection import Config, JaegerClient, AnomalyDetector
from dependency_analysis import DependencyGraphBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main-pipeline")

# Import causal inference modules (optional)
try:
    from causal_inference.root_cause_analysis import RootCauseAnalyzer
    CAUSAL_INFERENCE_AVAILABLE = True
except ImportError:
    CAUSAL_INFERENCE_AVAILABLE = False
    logger.warning("Causal inference modules not available. Install DoWhy with: pip install dowhy")


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
    anomalous_traces = anomaly_detector.detect(traces)
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


def run_causal_inference(traces_file: str, dependency_file: str, target_service: str, timestamp: str, 
                        train_model: bool = False) -> Optional[str]:
    """
    Step 3: Perform causal inference root cause analysis.
    
    Args:
        traces_file: Path to trace file
        dependency_file: Path to dependency file
        target_service: Target service for RCA
        timestamp: Timestamp for output files
        train_model: Whether to train a new model
        
    Returns: 
        path to RCA results file or None if failed
    """
    if not CAUSAL_INFERENCE_AVAILABLE:
        logger.warning("Causal inference not available - skipping RCA")
        return None
    
    logger.info("=== STEP 3: CAUSAL INFERENCE RCA ===")
    
    try:
        # Initialize analyzer
        analyzer = RootCauseAnalyzer()
        
        # Load data with caching support
        analyzer.load_data(traces_file, dependency_file, use_cache=not train_model)
        
        # Train model (will skip if loaded from cache)
        analyzer.train_causal_model(dependency_file=dependency_file)
        
        # Get available services
        available_services = list(analyzer.normal_data.columns) if analyzer.normal_data is not None else []
        
        # Check if target service exists
        if target_service not in available_services:
            logger.warning(f"Target service '{target_service}' not found in data.")
            if available_services:
                target_service = available_services[0]
                logger.info(f"Using '{target_service}' as target service instead")
            else:
                logger.warning("No services available for RCA")
                return None
        
        # Perform analyses if anomalous data is available
        results = {}
        if len(analyzer.anomalous_data) > 0:
            logger.info(f"Performing RCA for service: {target_service}")
            
            # Single outlier analysis
            outlier_result = analyzer.analyze_single_outlier(target_service, num_bootstrap=5)
            results['single_outlier'] = outlier_result
            
            # Distribution change analysis
            dist_result = analyzer.analyze_distribution_change(target_service, num_bootstrap=5)
            results['distribution_change'] = dist_result
            
            logger.info("RCA completed")
        else:
            logger.warning("No anomalous data found - skipping anomaly-specific RCA")
        
        # Add service summary and caching info
        results['service_summary'] = analyzer.get_service_summary()
        results['metadata'] = {
            'target_service': target_service,
            'timestamp': timestamp,
            'normal_samples': len(analyzer.normal_data) if analyzer.normal_data is not None else 0,
            'anomalous_samples': len(analyzer.anomalous_data) if analyzer.anomalous_data is not None else 0,
            'model_cached': not train_model,
            'cached_models': analyzer.list_cached_models()
        }
        
        # Save results
        output_dir = Path("output/rca")
        output_dir.mkdir(parents=True, exist_ok=True)
        rca_file = output_dir / f"rca_results_{target_service}_{timestamp}.json"
        
        with open(rca_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✓ RCA results saved: {rca_file}")
        return str(rca_file)
        
    except Exception as e:
        logger.error(f"Causal inference RCA failed: {e}")
        return None


def run_complete_pipeline(config, enable_causal_inference=False, rca_target_service=None, train_model=False):
    """Run the complete pipeline: trace collection + dependency analysis + causal inference RCA."""
    try:
        # Step 1: Collect traces from Jaeger
        traces_file, anomalous_file = collect_traces(config)
        
        if traces_file is None:
            logger.warning("No traces collected - pipeline stopped")
            return
        
        # Step 2: Analyze dependencies (conditional on causal inference settings)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = []
        
        if enable_causal_inference and rca_target_service:
            # Only run dependency analysis if we're training a new model
            if train_model:
                logger.info("🔄 Training mode: Generating dependency graph...")
                output_files = analyze_dependencies(traces_file, timestamp)
                dependency_file = output_files[0]  # Use the first output file (JSON summary)
            else:
                logger.info("⚡ Using cached model: Skipping dependency graph generation")
                # Look for existing dependency file
                dependency_dir = Path("output/analysis")
                dependency_file = None
                if dependency_dir.exists():
                    for file_path in dependency_dir.glob("dependency_analysis_*.json"):
                        dependency_file = str(file_path)
                        break
                
                if not dependency_file:
                    logger.warning("⚠️ No existing dependency file found. Running dependency analysis...")
                    output_files = analyze_dependencies(traces_file, timestamp)
                    dependency_file = output_files[0]
            
            # Step 3: Perform causal inference RCA
            rca_file = run_causal_inference(traces_file, dependency_file, rca_target_service, timestamp, train_model=train_model)
            if rca_file:
                output_files.append(rca_file)
        else:
            # If no causal inference, always run dependency analysis
            logger.info("📊 Running dependency analysis...")
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
  python main.py --causal-inference           # Include causal inference RCA
  python main.py --rca-target frontend        # Specify target service for RCA
  python main.py --train-model --causal-inference  # Train new model and do RCA
        '''
    )
    
    parser.add_argument('--continuous', action='store_true',
                       help='Run in continuous mode')
    parser.add_argument('--interval', type=int, default=300,
                       help='Interval between collections in continuous mode (seconds, default: 300)')
    parser.add_argument('--analyze-only', type=str, metavar='TRACES_FILE',
                       help='Skip trace collection and only analyze existing traces file')
    parser.add_argument('--causal-inference', action='store_true',
                       help='Perform causal inference analysis (RCA)')
    parser.add_argument('--rca-target', type=str, metavar='SERVICE',
                       help='Target service for causal inference RCA')
    parser.add_argument('--train-model', action='store_true',
                       help='Train a new causal model (default: use existing model if available)')
    
    args = parser.parse_args()
    
    # Analyze-only mode (no trace collection)
    if args.analyze_only:
        if not os.path.exists(args.analyze_only):
            logger.error(f"Traces file not found: {args.analyze_only}")
            return 1
            
        try:
            # Load traces for anomaly detection
            with open(args.analyze_only, 'r') as f:
                data = json.load(f)
                
            # Handle different trace file formats
            if isinstance(data, dict) and 'data' in data:
                traces = data['data']
            elif isinstance(data, list):
                traces = data
            else:
                traces = [data]
            
            # Initialize anomaly detector and run detection
            config = Config()
            anomaly_detector = AnomalyDetector(config.LATENCY_PERCENTILE)
            
            logger.info("Detecting anomalous traces...")
            anomalous_traces = anomaly_detector.detect(traces)
            logger.info(f"Found {len(anomalous_traces)} anomalous traces")
            
            # Save anomalous traces
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("output/traces")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            anomalous_file = output_dir / f"anomalous_traces_{timestamp}.json"
            with open(anomalous_file, 'w') as f:
                json.dump(anomalous_traces, f, indent=2)
            logger.info(f"Saved {len(anomalous_traces)} anomalous traces to: {anomalous_file}")
            
            # Optional: Run causal inference if requested
            if args.causal_inference and args.rca_target:
                if CAUSAL_INFERENCE_AVAILABLE:
                    logger.info("🧠 Running causal inference analysis...")
                    
                    # Only run dependency analysis if we're training a new model
                    if args.train_model:
                        logger.info("🔄 Training mode: Generating dependency graph...")
                        output_files = analyze_dependencies(args.analyze_only)
                        
                        # Find the dependency JSON file from output_files
                        dependency_file = None
                        for file_path in output_files:
                            if file_path.endswith('dependency_analysis_*.json') or 'dependency_analysis' in file_path:
                                dependency_file = file_path
                                break
                    else:
                        logger.info("⚡ Using cached model: Skipping dependency graph generation")
                        # Look for existing dependency file
                        dependency_dir = Path("output/analysis")
                        dependency_file = None
                        if dependency_dir.exists():
                            for file_path in dependency_dir.glob("dependency_analysis_*.json"):
                                dependency_file = str(file_path)
                                break
                        
                        if not dependency_file:
                            logger.warning("⚠️ No existing dependency file found. Running dependency analysis...")
                            output_files = analyze_dependencies(args.analyze_only)
                            for file_path in output_files:
                                if file_path.endswith('dependency_analysis_*.json') or 'dependency_analysis' in file_path:
                                    dependency_file = file_path
                                    break
                    
                    if dependency_file:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        rca_file = run_causal_inference(args.analyze_only, dependency_file, args.rca_target, timestamp, train_model=args.train_model)
                        if rca_file:
                            logger.info(f"✅ Causal inference completed! Results saved to: {rca_file}")
                        else:
                            logger.warning("⚠️ Causal inference failed or returned no results")
                    else:
                        logger.error("❌ Could not find dependency analysis file for causal inference")
                else:
                    logger.error("❌ Causal inference not available. Install DoWhy with: pip install dowhy")
            elif args.causal_inference and not args.rca_target:
                logger.warning("⚠️ Causal inference requested but no target service specified. Use --rca-target <service>")
            elif not args.causal_inference:
                # If no causal inference, always run dependency analysis
                logger.info("📊 Running dependency analysis...")
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
    
    # Configure causal inference settings
    enable_causal_inference = args.causal_inference
    rca_target_service = args.rca_target or 'frontend'  # Default target service
    
    if enable_causal_inference:
        if CAUSAL_INFERENCE_AVAILABLE:
            training_status = "enabled" if args.train_model else "disabled"
            logger.info(f"🧠 Causal inference RCA enabled for service: {rca_target_service} (model training: {training_status})")
        else:
            logger.warning("🚫 Causal inference requested but DoWhy not available")
            enable_causal_inference = False
    
    if args.continuous:
        logger.info(f"🔄 Continuous mode enabled (interval: {args.interval}s)")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                run_complete_pipeline(config, enable_causal_inference, rca_target_service, args.train_model)
                logger.info("⏳ Waiting until next collection...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("🛑 Pipeline stopped by user")
    else:
        logger.info("🎯 Single run mode")
        run_complete_pipeline(config, enable_causal_inference, rca_target_service, args.train_model)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())