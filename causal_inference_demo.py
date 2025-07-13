#!/usr/bin/env python3
"""
Causal Inference Pipeline for Root Cause Analysis

This script demonstrates the complete pipeline for performing root cause analysis
on microservice latency data using causal inference techniques.

Usage:
    python causal_inference_demo.py [--target-service SERVICE] [--output-dir DIR]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.causal_inference.root_cause_analysis import RootCauseAnalyzer
from src.causal_inference.data_preparation import process_traces_for_causal_analysis

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('causal_inference.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Run causal inference root cause analysis')
    parser.add_argument('--target-service', default='frontend', 
                       help='Target service to analyze for anomalies')
    parser.add_argument('--output-dir', default='output/rca_results',
                       help='Directory to save results')
    parser.add_argument('--trace-file', default='output/traces/collected_traces_20250713_230056.json',
                       help='Path to trace data file')
    parser.add_argument('--dependency-file', default='output/analysis/dependency_analysis_20250713_230114.json',
                       help='Path to dependency analysis file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting causal inference root cause analysis")
    
    try:
        # Check if required files exist
        trace_file = Path(args.trace_file)
        dependency_file = Path(args.dependency_file)
        
        if not trace_file.exists():
            logger.error(f"Trace file not found: {trace_file}")
            return 1
        
        if not dependency_file.exists():
            logger.error(f"Dependency file not found: {dependency_file}")
            return 1
        
        # Initialize analyzer
        analyzer = RootCauseAnalyzer()
        
        # Load data
        logger.info("Loading trace and dependency data...")
        analyzer.load_data(str(trace_file), str(dependency_file))
        
        # Print service summary
        summary = analyzer.get_service_summary()
        logger.info(f"Service Summary:")
        logger.info(f"  Total services: {summary['total_services']}")
        logger.info(f"  Causal edges: {summary['total_causal_edges']}")
        logger.info(f"  Root services: {summary['root_services']}")
        logger.info(f"  Leaf services: {summary['leaf_services']}")
        
        # Check if target service exists
        available_services = list(analyzer.normal_data.columns) if analyzer.normal_data is not None else []
        if args.target_service not in available_services:
            logger.warning(f"Target service '{args.target_service}' not found in data.")
            logger.info(f"Available services: {available_services}")
            if available_services:
                args.target_service = available_services[0]
                logger.info(f"Using '{args.target_service}' as target service instead")
            else:
                logger.error("No services available for analysis")
                return 1
        
        # Train causal model
        logger.info("Training causal model...")
        analyzer.train_causal_model()
        
        # Perform analyses if anomalous data is available
        if len(analyzer.anomalous_data) > 0:
            logger.info(f"Performing root cause analysis for service: {args.target_service}")
            
            # Single outlier analysis
            outlier_result = analyzer.analyze_single_outlier(args.target_service)
            logger.info(f"Single outlier analysis completed:")
            logger.info(f"  Outlier magnitude: {outlier_result['outlier_magnitude']:.2f}ms")
            
            # Distribution change analysis
            dist_result = analyzer.analyze_distribution_change(args.target_service)
            logger.info(f"Distribution change analysis completed:")
            logger.info(f"  Change magnitude: {dist_result['change_magnitude']:.2f}ms")
            
            # Print top attributions
            print("\n" + "="*60)
            print("ROOT CAUSE ANALYSIS RESULTS")
            print("="*60)
            
            print(f"\nTarget Service: {args.target_service}")
            print(f"Outlier Magnitude: {outlier_result['outlier_magnitude']:.2f}ms")
            print(f"Distribution Change: {dist_result['change_magnitude']:.2f}ms")
            
            print("\nTop Root Cause Attributions (Single Outlier):")
            sorted_attribs = sorted(outlier_result['attributions'].items(), 
                                  key=lambda x: abs(float(x[1])), reverse=True)[:5]
            for service, score in sorted_attribs:
                uncertainty = outlier_result['uncertainties'][service]
                score_val = float(score) if hasattr(score, '__float__') else score
                unc_range = (float(uncertainty[1]) - float(uncertainty[0])) / 2
                print(f"  {service}: {score_val:.4f} (±{unc_range:.4f})")
            
            print("\nTop Root Cause Attributions (Distribution Change):")
            sorted_dist_attribs = sorted(dist_result['attributions'].items(), 
                                       key=lambda x: abs(float(x[1])), reverse=True)[:5]
            for service, score in sorted_dist_attribs:
                uncertainty = dist_result['uncertainties'][service]
                score_val = float(score) if hasattr(score, '__float__') else score
                unc_range = (float(uncertainty[1]) - float(uncertainty[0])) / 2
                print(f"  {service}: {score_val:.4f} (±{unc_range:.4f})")
            
            # Example intervention simulation
            print("\nSimulating intervention...")
            top_cause = sorted_attribs[0][0] if sorted_attribs else args.target_service
            
            # Simulate reducing latency of top cause by 50%
            interventions = {top_cause: lambda x: x * 0.5}
            intervention_result = analyzer.simulate_intervention(interventions)
            
            baseline_latency = float(intervention_result['baseline_means'][args.target_service])
            intervention_latency = float(intervention_result['intervention_results'][args.target_service])
            improvement = baseline_latency - intervention_latency
            
            print(f"Intervention: Reduce {top_cause} latency by 50%")
            print(f"  Baseline {args.target_service} latency: {baseline_latency:.2f}ms")
            print(f"  After intervention: {intervention_latency:.2f}ms")
            print(f"  Improvement: {improvement:.2f}ms")
            
            # Visualize results
            try:
                analyzer.visualize_results('single_outlier', args.output_dir)
                analyzer.visualize_results('distribution_change', args.output_dir)
            except Exception as e:
                logger.warning(f"Could not create visualizations: {e}")
        
        else:
            logger.warning("No anomalous data found - skipping anomaly-specific analyses")
            
            # Still show service statistics
            print("\n" + "="*60)
            print("SERVICE LATENCY STATISTICS")
            print("="*60)
            
            for service, stats in summary['service_latency_stats'].items():
                print(f"\n{service}:")
                print(f"  Mean: {stats['mean']:.2f}ms")
                print(f"  Std:  {stats['std']:.2f}ms")
                print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}ms")
        
        # Save results
        results_file = analyzer.save_results(args.output_dir)
        logger.info(f"Results saved to: {results_file}")
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in causal inference analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
