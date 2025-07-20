#!/usr/bin/env python3
"""
Causal Inference RCA System - Training and Analysis Pipeline

This system provides four distinct operation modes:

TRAINING MODES:
1. Train model with Jaeger backend traces
   - Fetches fresh traces from Jaeger
   - Builds dependency graph and trains causal model
   - Saves model for later RCA analysis

2. Train model with local trace file  
   - Uses existing trace file for training
   - Builds dependency graph and trains causal model
   - Saves model for later RCA analysis

RCA ANALYSIS MODES:
3. RCA with Jaeger traces
   - Collects fresh traces from Jaeger
   - Detects anomalous traces
   - Uses existing trained model to perform RCA

4. RCA with existing traces
   - Uses existing trace file
   - Detects anomalous traces
   - Uses existing trained model to perform RCA

Usage Examples:
    # Training modes
    python main.py --train-jaeger --jaeger-url http://localhost:16686
    python main.py --train-file output/traces/traces.json
    
    # RCA modes
    python main.py --rca-jaeger --jaeger-url http://localhost:16686 --model-path output/models/model.pkl --target-service frontend_proc
    python main.py --rca-file output/traces/traces.json --model-path output/models/model.pkl --target-service frontend_proc
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
        description='Causal Inference RCA System - Training and Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
TRAINING MODES:
  python main.py --train-jaeger --jaeger-url http://localhost:16686
      Train model by fetching traces from Jaeger backend
  
  python main.py --train-file /path/to/traces.json  
      Train model using existing local trace file

RCA ANALYSIS MODES:
  python main.py --rca-jaeger --model-path output/models/model.pkl
      Perform RCA by collecting fresh traces from Jaeger
      
  python main.py --rca-file /path/to/traces.json --model-path output/models/model.pkl
      Perform RCA on existing trace file

LEGACY MODE:
  python main.py --continuous --interval 600  # Continuous collection mode
        '''
    )
    
    # Training modes
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument('--train-jaeger', action='store_true',
                           help='Train model by fetching traces from Jaeger')
    train_group.add_argument('--train-file', type=str, metavar='TRACES_FILE',
                           help='Train model using existing local trace file')
    
    # RCA modes  
    rca_group = parser.add_mutually_exclusive_group()
    rca_group.add_argument('--rca-jaeger', action='store_true',
                         help='Perform RCA by collecting traces from Jaeger')
    rca_group.add_argument('--rca-file', type=str, metavar='TRACES_FILE',
                         help='Perform RCA on existing trace file')
    
    # Required arguments for specific modes
    parser.add_argument('--jaeger-url', type=str, 
                       help='Jaeger URL (required for --train-jaeger and --rca-jaeger)')
    parser.add_argument('--model-path', type=str,
                       help='Path to saved model (required for RCA modes)')
    parser.add_argument('--target-service', type=str,
                       help='Target service for RCA analysis')
    
    # Legacy continuous mode
    parser.add_argument('--continuous', action='store_true',
                       help='Run in continuous collection mode (legacy)')
    parser.add_argument('--interval', type=int, default=300,
                       help='Interval between collections in continuous mode (seconds, default: 300)')
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.train_jaeger or args.rca_jaeger:
        if not args.jaeger_url:
            logger.error("❌ --jaeger-url is required for Jaeger-based operations")
            return 1
    
    if args.rca_jaeger or args.rca_file:
        if not args.model_path:
            logger.error("❌ --model-path is required for RCA operations")
            return 1
        if not args.target_service:
            logger.error("❌ --target-service is required for RCA operations")
            return 1
    
    # Execute based on selected mode
    try:
        if args.train_jaeger:
            logger.info("🚀 TRAINING MODE: Jaeger Backend")
            return train_model_from_jaeger(args.jaeger_url)
            
        elif args.train_file:
            logger.info("🚀 TRAINING MODE: Local File")
            return train_model_from_file(args.train_file)
            
        elif args.rca_jaeger:
            logger.info("🚀 RCA MODE: Jaeger Backend")
            return perform_rca_from_jaeger(args.jaeger_url, args.model_path, args.target_service)
            
        elif args.rca_file:
            logger.info("🚀 RCA MODE: Local File")
            return perform_rca_from_file(args.rca_file, args.model_path, args.target_service)
            
        elif args.continuous:
            logger.info("🚀 LEGACY MODE: Continuous Collection")
            return run_legacy_continuous_mode(args)
            
        else:
            logger.error("❌ No operation mode specified. Use --help for usage information.")
            parser.print_help()
            return 1
            
    except Exception as e:
        logger.error(f"❌ Operation failed: {e}")
        return 1


def train_model_from_jaeger(jaeger_url: str) -> int:
    """
    Training Mode 1: Train model by fetching traces from Jaeger backend.
    
    Args:
        jaeger_url: Jaeger URL endpoint
        
    Returns:
        0 if successful, 1 if failed
    """
    logger.info(f"📡 Connecting to Jaeger at: {jaeger_url}")
    
    # Configure for Jaeger collection
    config = Config()
    # Override Jaeger URL from config
    host, port = jaeger_url.replace('http://', '').split(':')
    config.JAEGER_QUERY_HOST = host
    config.JAEGER_QUERY_PORT = port
    
    # Step 1: Collect traces from Jaeger
    traces_file, anomalous_file = collect_traces(config)
    
    if traces_file is None:
        logger.error("❌ Failed to collect traces from Jaeger")
        return 1
    
    # Step 2: Build dependency graph and train model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Generate dependency analysis
        logger.info("📊 Analyzing dependencies and training causal model...")
        output_files = analyze_dependencies(traces_file, timestamp)
        dependency_file = output_files[0]  # JSON summary file
        
        # Train causal model
        analyzer = RootCauseAnalyzer()
        analyzer.load_data(traces_file, dependency_file, use_cache=False)
        model_path = analyzer.train_causal_model(dependency_file=dependency_file)
        
        logger.info("✅ MODEL TRAINING COMPLETED")
        logger.info(f"📊 Traces processed: {traces_file}")
        logger.info(f"🔗 Dependencies: {dependency_file}")
        logger.info(f"🧠 Model saved: {model_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        return 1


def train_model_from_file(traces_file: str) -> int:
    """
    Training Mode 2: Train model using existing local trace file.
    
    Args:
        traces_file: Path to existing trace file
        
    Returns:
        0 if successful, 1 if failed
    """
    logger.info(f"📁 Loading traces from file: {traces_file}")
    
    if not os.path.exists(traces_file):
        logger.error(f"❌ Trace file not found: {traces_file}")
        return 1
    
    try:
        # Perform anomaly detection on existing traces
        with open(traces_file, 'r') as f:
            data = json.load(f)
            
        # Handle different trace file formats
        if isinstance(data, dict) and 'data' in data:
            traces = data['data']
        elif isinstance(data, list):
            traces = data
        else:
            traces = [data]
        
        logger.info(f"📊 Processing {len(traces)} traces from file")
        
        # Initialize anomaly detector and run detection
        config = Config()
        anomaly_detector = AnomalyDetector(config.LATENCY_PERCENTILE)
        
        logger.info("🔍 Detecting anomalous traces...")
        anomalous_traces = anomaly_detector.detect(traces)
        logger.info(f"Found {len(anomalous_traces)} anomalous traces")
        
        # Save anomalous traces
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output/traces")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        anomalous_file = output_dir / f"anomalous_traces_{timestamp}.json"
        with open(anomalous_file, 'w') as f:
            json.dump(anomalous_traces, f, indent=2)
        logger.info(f"💾 Saved anomalous traces: {anomalous_file}")
        
        # Generate dependency analysis and train model
        logger.info("📊 Analyzing dependencies and training causal model...")
        output_files = analyze_dependencies(traces_file, timestamp)
        dependency_file = output_files[0]  # JSON summary file
        
        # Train causal model
        analyzer = RootCauseAnalyzer()
        analyzer.load_data(traces_file, dependency_file, use_cache=False)
        analyzer.train_causal_model(dependency_file=dependency_file)
        model_path = analyzer.get_model_path(dependency_file)
        
        logger.info("✅ MODEL TRAINING COMPLETED")
        logger.info(f"📁 Input traces: {traces_file}")
        logger.info(f"🚨 Anomalous traces: {anomalous_file}")
        logger.info(f"🔗 Dependencies: {dependency_file}")
        logger.info(f"🧠 Model saved: {model_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        return 1


def perform_rca_from_jaeger(jaeger_url: str, model_path: str, target_service: str) -> int:
    """
    RCA Mode 1: Perform RCA by collecting traces from Jaeger.
    
    Args:
        jaeger_url: Jaeger URL endpoint
        model_path: Path to saved model
        target_service: Target service for RCA
        
    Returns:
        0 if successful, 1 if failed
    """
    logger.info(f"🎯 RCA Analysis - Target: {target_service}")
    logger.info(f"📡 Jaeger URL: {jaeger_url}")
    logger.info(f"🧠 Model path: {model_path}")
    
    # Validate model exists
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        return 1
    
    # Configure for Jaeger collection
    config = Config()
    host, port = jaeger_url.replace('http://', '').split(':')
    config.JAEGER_QUERY_HOST = host
    config.JAEGER_QUERY_PORT = port
    
    try:
        # Step 1: Collect fresh traces from Jaeger
        traces_file, anomalous_file = collect_traces(config)
        
        if traces_file is None:
            logger.error("❌ Failed to collect traces from Jaeger")
            return 1
        
        # Step 2: Perform RCA using cached model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Find existing dependency file (required for RCA)
        dependency_dir = Path("output/analysis")
        dependency_file = None
        if dependency_dir.exists():
            for file_path in dependency_dir.glob("dependency_analysis_*.json"):
                dependency_file = str(file_path)
                break
        
        if not dependency_file:
            logger.warning("⚠️ No existing dependency file found. Generating one...")
            output_files = analyze_dependencies(traces_file, timestamp)
            dependency_file = output_files[0]
        
        # Perform RCA with cached model
        rca_file = run_causal_inference(traces_file, dependency_file, target_service, timestamp, train_model=False)
        
        if rca_file:
            logger.info("✅ RCA ANALYSIS COMPLETED")
            logger.info(f"📊 Fresh traces: {traces_file}")
            logger.info(f"🚨 Anomalies: {anomalous_file}")
            logger.info(f"🎯 RCA results: {rca_file}")
            return 0
        else:
            logger.error("❌ RCA analysis failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ RCA analysis failed: {e}")
        return 1


def perform_rca_from_file(traces_file: str, model_path: str, target_service: str) -> int:
    """
    RCA Mode 2: Perform RCA on existing trace file.
    
    Args:
        traces_file: Path to existing trace file
        model_path: Path to saved model  
        target_service: Target service for RCA
        
    Returns:
        0 if successful, 1 if failed
    """
    logger.info(f"🎯 RCA Analysis - Target: {target_service}")
    logger.info(f"📁 Traces file: {traces_file}")
    logger.info(f"🧠 Model path: {model_path}")
    
    # Validate inputs
    if not os.path.exists(traces_file):
        logger.error(f"❌ Trace file not found: {traces_file}")
        return 1
        
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        return 1
    
    try:
        # Step 1: Detect anomalous traces
        with open(traces_file, 'r') as f:
            data = json.load(f)
            
        # Handle different trace file formats
        if isinstance(data, dict) and 'data' in data:
            traces = data['data']
        elif isinstance(data, list):
            traces = data
        else:
            traces = [data]
        
        logger.info(f"📊 Processing {len(traces)} traces")
        
        # Run anomaly detection
        config = Config()
        anomaly_detector = AnomalyDetector(config.LATENCY_PERCENTILE)
        
        logger.info("🔍 Detecting anomalous traces...")
        anomalous_traces = anomaly_detector.detect(traces)
        logger.info(f"Found {len(anomalous_traces)} anomalous traces")
        
        # Save anomalous traces
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output/traces")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        anomalous_file = output_dir / f"anomalous_traces_{timestamp}.json"
        with open(anomalous_file, 'w') as f:
            json.dump(anomalous_traces, f, indent=2)
        
        # Step 2: Perform RCA using cached model
        # Find existing dependency file
        dependency_dir = Path("output/analysis")
        dependency_file = None
        if dependency_dir.exists():
            for file_path in dependency_dir.glob("dependency_analysis_*.json"):
                dependency_file = str(file_path)
                break
        
        if not dependency_file:
            logger.warning("⚠️ No existing dependency file found. Generating one...")
            output_files = analyze_dependencies(traces_file, timestamp)
            dependency_file = output_files[0]
        
        # Perform RCA with cached model
        rca_file = run_causal_inference(traces_file, dependency_file, target_service, timestamp, train_model=False)
        
        if rca_file:
            logger.info("✅ RCA ANALYSIS COMPLETED")
            logger.info(f"📁 Input traces: {traces_file}")
            logger.info(f"🚨 Anomalies: {anomalous_file}")
            logger.info(f"🎯 RCA results: {rca_file}")
            return 0
        else:
            logger.error("❌ RCA analysis failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ RCA analysis failed: {e}")
        return 1


def run_legacy_continuous_mode(args) -> int:
    """
    Legacy continuous collection mode.
    """
    config = Config()
    logger.info("🚀 Starting Legacy Continuous Mode")
    logger.info(f"📡 Jaeger endpoint: {config.jaeger_endpoint}")
    logger.info(f"🔄 Interval: {args.interval}s")
    logger.info("Press Ctrl+C to stop")
    
    try:
        while True:
            run_complete_pipeline(config, False, None, False)
            logger.info("⏳ Waiting until next collection...")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("🛑 Pipeline stopped by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Continuous mode failed: {e}")
        return 1


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


if __name__ == "__main__":
    sys.exit(main())