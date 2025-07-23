# Causal Inference RCA System

This project provides a complete pipeline for root cause analysis (RCA) in distributed microservice systems using causal inference. It supports both synthetic trace generation and real trace ingestion from Jaeger, enabling users to simulate, analyze, and visualize performance issues in complex service topologies.

## Features
- **Synthetic Trace Generation:** Simulate realistic e-commerce microservice topologies with configurable performance bottlenecks.
- **Jaeger Integration:** Collect and analyze real distributed traces from a Jaeger backend.
- **Causal Model Training:** Build causal models to understand dependencies and propagation of latency or failures.
- **Root Cause Analysis:** Identify the most likely root causes of anomalies in service traces.
- **Visualization:** Generate clear visualizations of RCA results for rapid diagnosis.

## Project Structure
- `src/` - Core source code for trace generation, causal inference, and analysis
- `output/` - Generated traces, models, analysis results, and visualizations
- `main.py` - Command-line interface for training and RCA
- `api_server.py` - REST API for remote operation
- `Dockerfile` - Containerization for easy deployment

## Getting Started
For step-by-step deployment and usage instructions, including Docker-based workflows, see the [Docker Deployment Guide](./DOCKER_DEPLOYMENT_GUIDE.md).

