"""
Dependency Analysis Package

This package provides functionality for analyzing service dependencies 
from distributed traces and building dependency graphs.
"""

from .dependency_graph import DependencyGraphBuilder, analyze_trace_dependencies

__all__ = ['DependencyGraphBuilder', 'analyze_trace_dependencies']
