#!/usr/bin/env python3
"""
Attribution Scores Visualization Tool
Displays service attribution scores with uncertainties from RCA results
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def parse_attribution_value(attr_str):
    """Parse attribution string that might be wrapped in brackets"""
    if isinstance(attr_str, str):
        # Remove brackets and parse
        cleaned = attr_str.strip('[]')
        return float(cleaned)
    return float(attr_str)

def parse_uncertainty_bounds(uncertainty_list):
    """Parse uncertainty bounds from list format"""
    if isinstance(uncertainty_list, list) and len(uncertainty_list) == 2:
        lower = parse_attribution_value(uncertainty_list[0])
        upper = parse_attribution_value(uncertainty_list[1])
        return lower, upper
    return None, None

def load_rca_results(filepath):
    """Load RCA results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_attribution_plot(rca_data, analysis_type='single_outlier'):
    """Create attribution scores plot with uncertainties"""
    
    # Extract data based on analysis type
    if analysis_type not in rca_data:
        print(f"Analysis type '{analysis_type}' not found in results")
        return None
    
    analysis_data = rca_data[analysis_type]
    target_service = analysis_data['target_service']
    
    # Get attributions and uncertainties
    attributions = analysis_data.get('attributions', {})
    uncertainties = analysis_data.get('uncertainties', {})
    
    # Parse attribution scores
    services = []
    attr_values = []
    error_lower = []
    error_upper = []
    
    for service, attr_val in attributions.items():
        services.append(service)
        attr_score = parse_attribution_value(attr_val)
        attr_values.append(attr_score)
        
        # Parse uncertainty bounds
        if service in uncertainties:
            lower, upper = parse_uncertainty_bounds(uncertainties[service])
            if lower is not None and upper is not None:
                # Calculate error bars relative to the attribution value
                error_lower.append(abs(attr_score - lower))
                error_upper.append(abs(upper - attr_score))
            else:
                error_lower.append(0)
                error_upper.append(0)
        else:
            error_lower.append(0)
            error_upper.append(0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort services by attribution magnitude for better visualization
    sorted_indices = np.argsort(np.abs(attr_values))[::-1]
    services_sorted = [services[i] for i in sorted_indices]
    attr_values_sorted = [attr_values[i] for i in sorted_indices]
    error_lower_sorted = [error_lower[i] for i in sorted_indices]
    error_upper_sorted = [error_upper[i] for i in sorted_indices]
    
    # Create bar plot with error bars
    bars = ax.barh(range(len(services_sorted)), attr_values_sorted, 
                   xerr=[error_lower_sorted, error_upper_sorted],
                   capsize=5, alpha=0.7)
    
    # Color bars - red for positive, blue for negative
    for i, (bar, value) in enumerate(zip(bars, attr_values_sorted)):
        if value > 0:
            bar.set_color('red')
        else:
            bar.set_color('blue')
    
    # Highlight the target service
    target_index = services_sorted.index(target_service) if target_service in services_sorted else -1
    if target_index >= 0:
        bars[target_index].set_color('darkred' if attr_values_sorted[target_index] > 0 else 'darkblue')
        bars[target_index].set_alpha(1.0)
    
    # Customize the plot
    ax.set_yticks(range(len(services_sorted)))
    ax.set_yticklabels(services_sorted)
    ax.set_xlabel('Attribution Score')
    ax.set_title(f'{target_service.upper()} - Attribution Scores with Uncertainties', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Add text annotations for high attribution scores
    for i, (service, value) in enumerate(zip(services_sorted, attr_values_sorted)):
        if abs(value) > max(abs(v) for v in attr_values_sorted) * 0.1:  # Show if > 10% of max
            ax.text(value + (0.05 if value > 0 else -0.05), i, 
                   f'{value:.3f}', 
                   va='center', ha='left' if value > 0 else 'right',
                   fontweight='bold', fontsize=10)
    
    # Add metadata info
    metadata_text = f"Analysis: {analysis_type.replace('_', ' ').title()}\n"
    if 'outlier_magnitude' in analysis_data:
        metadata_text += f"Outlier Magnitude: {analysis_data['outlier_magnitude']:.2f}\n"
    if 'change_magnitude' in analysis_data:
        metadata_text += f"Change Magnitude: {analysis_data['change_magnitude']:.2f}\n"
    
    # Find which service was slowed down (highest attribution score - this is the root cause)
    if services_sorted and attr_values_sorted:
        slowed_service = (services_sorted[0], attr_values_sorted[0])  # Already sorted by highest magnitude
        metadata_text += f"Root Cause Service: {slowed_service[0]} (attr: {slowed_service[1]:.3f})"
    
    ax.text(0.02, 0.98, metadata_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize RCA attribution scores')
    parser.add_argument('--file', '-f', required=True, help='Path to RCA results JSON file')
    parser.add_argument('--type', '-t', default='single_outlier', 
                       choices=['single_outlier', 'distribution_change'],
                       help='Analysis type to visualize')
    parser.add_argument('--save', '-s', help='Save plot to file (optional)')
    parser.add_argument('--show', action='store_true', help='Display plot')
    
    args = parser.parse_args()
    
    # Load and visualize results
    try:
        rca_data = load_rca_results(args.file)
        fig = create_attribution_plot(rca_data, args.type)
        
        if fig:
            if args.save:
                fig.savefig(args.save, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {args.save}")
            
            if args.show:
                plt.show()
            
            if not args.save and not args.show:
                # Default: save with auto-generated name
                target_service = rca_data[args.type]['target_service']
                timestamp = rca_data['metadata']['timestamp']
                filename = f"attribution_plot_{target_service}_{timestamp}.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {filename}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
