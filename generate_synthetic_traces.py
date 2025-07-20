#!/usr/bin/env python3
"""
Synthetic Trace Generator CLI

Simple command-line interface for generating synthetic OTEL traces
with controlled latency anomalies.

Usage:
    python generate_synthetic_traces.py --num-traces 1000 --slow-service payment_svc
    python generate_synthetic_traces.py --list-services
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.synthetic_traces import TraceGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic OTEL traces with controlled latency anomalies"
    )
    
    parser.add_argument(
        '--num-traces', 
        type=int, 
        default=1000,
        help='Number of traces to generate (default: 1000)'
    )
    
    parser.add_argument(
        '--slow-service',
        type=str,
        help='Which service to slow down in anomalous traces'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output file path (default: auto-generated)'
    )
    
    parser.add_argument(
        '--list-services',
        action='store_true',
        help='List available services and topology'
    )
    
    args = parser.parse_args()
    
    generator = TraceGenerator()
    
    if args.list_services:
        print("Available Services:")
        services = generator.get_available_services()
        for service in services:
            print(f"  - {service}")
        print()
        print("Service Topology:")
        print(generator.get_topology_info())
        return
    
    if not args.slow_service:
        print("Error: --slow-service is required")
        print("Use --list-services to see available options")
        sys.exit(1)
    
    try:
        output_file = generator.generate_traces(
            num_traces=args.num_traces,
            slow_service=args.slow_service,
            output_file=args.output_file
        )
        
        print(f"\\n🎉 Success!")
        print(f"You can now test with: python main.py rca-file --traces {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
