#!/usr/bin/env python3
"""
CLI Module - Handles command-line interface and argument parsing
"""

import argparse
import logging
import sys
from typing import Optional

logger = logging.getLogger("cli")


class CLI:
    """Command Line Interface handler"""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def parse_args(self, args: Optional[list] = None):
        """Parse command line arguments"""
        return self.parser.parse_args(args)
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser"""
        parser = argparse.ArgumentParser(
            description='Causal Inference RCA System - Training and Analysis Pipeline',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
TRAINING MODES:
  python main.py --train-file /path/to/traces.json  
      Train model using existing local trace file

RCA ANALYSIS MODES:
  python main.py --rca-file /path/to/traces.json --model-path output/models/model.pkl --target-service SERVICE
      Perform RCA on existing trace file
            '''
        )
        
        # Training modes
        train_group = parser.add_mutually_exclusive_group()
        train_group.add_argument('--train-file', type=str, metavar='TRACES_FILE',
                               help='Train model using existing local trace file')
        
        # RCA modes  
        rca_group = parser.add_mutually_exclusive_group()
        rca_group.add_argument('--rca-file', type=str, metavar='TRACES_FILE',
                             help='Perform RCA on existing trace file')
        
        # Required arguments for specific modes
        parser.add_argument('--model-path', type=str,
                           help='Path to saved model (required for RCA modes)')
        parser.add_argument('--target-service', type=str,
                           help='Target service for RCA analysis')
        
        return parser
    
    def validate_args(self, args) -> bool:
        """Validate argument combinations"""
        if args.rca_file:
            if not args.model_path:
                logger.error("--model-path is required for RCA operations")
                return False
            if not args.target_service:
                logger.error("--target-service is required for RCA operations")
                return False
        
        if not args.train_file and not args.rca_file:
            logger.error("No operation mode specified. Use --help for usage information.")
            self.parser.print_help()
            return False
        
        return True
