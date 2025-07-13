# Trace Collection Service

A lightweight service for collecting and analyzing traces from Jaeger.

## Features

- **Periodic Trace Collection**: Automatically fetches traces from Jaeger at configurable intervals
- **Anomaly Detection**: Identifies traces with abnormal latencies using percentile-based thresholds
- **Data Export**: Saves collected traces and anomalies to JSON files for further analysis
- **Configurable**: Environment variable-based configuration for different deployment scenarios

## Quick Start

### Prerequisites
- Python 3.8+
- Access to a Jaeger instance

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
Set environment variables to configure the service:

```bash
export JAEGER_QUERY_HOST=localhost      # Jaeger host
export JAEGER_QUERY_PORT=16686          # Jaeger port
export SERVICE_NAME=your-service-name   # Service to monitor
export POLL_INTERVAL=3600               # Collection interval (seconds)
export PAST_INTERVAL=3600               # Time window to collect (seconds)
export LATENCY_PERCENTILE=99            # Anomaly threshold percentile
export MAX_TRACES=10000                 # Maximum traces per collection
export OUTPUT_DIR=output/traces         # Directory for output files (optional)
export AUTO_ANALYZE=true                # Enable automatic dependency analysis (optional)
```

### Running the Service
```bash
python trace_collector.py
```

### Continuous Pipeline Mode

The trace collector can automatically run dependency analysis after each trace collection cycle:

```bash
# Enable automatic dependency analysis (default)
export AUTO_ANALYZE=true
python trace_collector.py
```

This creates a continuous pipeline:
1. **Collect traces** from Jaeger every `POLL_INTERVAL` seconds
2. **Save traces** to `output/traces/collected_traces_*.json`
3. **Detect anomalies** and save to `output/traces/anomalous_traces_*.json`
4. **Automatically analyze dependencies** and save results to `output/analysis/`
5. **Repeat** the cycle

Pipeline outputs include:
- Service dependency graphs (PNG visualizations)
- Graph data exports (JSON, DOT formats)
- Adjacency lists and summaries

### Analyzing Service Dependencies
To analyze service dependencies from collected traces:
```bash
# Analyze dependencies from a trace file
python analyze_dependencies.py traces.json

# With visualization (requires matplotlib)
python analyze_dependencies.py traces.json --visualize

# Export graph to different formats
python analyze_dependencies.py traces.json --export-json deps.json --export-graphml deps.graphml
```

## Architecture

```
trace_collector.py          # Main trace collection application
analyze_dependencies.py     # Dependency graph analysis tool
src/
├── trace_collection/       # Trace collection and anomaly detection
│   ├── config.py           # Configuration management
│   ├── jaeger_client.py    # Jaeger API client
│   └── anomaly_detection.py # Anomaly detection logic
└── dependency_analysis/    # Service dependency analysis
    ├── dependency_graph.py # Dependency graph builder
    └── analyze_dependencies.py # Analysis script
traces.json                 # Sample trace data
```

## Output

The service generates timestamped JSON files in the configured output directory (default: `output/traces/`):
- `collected_traces_YYYYMMDD_HHMMSS.json` - All collected traces
- `anomalous_traces_YYYYMMDD_HHMMSS.json` - Traces identified as anomalous

Analysis outputs are saved to `output/analysis/`:
- `*_dependency_graph.png` - Visualization images
- `*_dependency_graph.json` - Graph data in JSON format
- `*_dependency_graph.dot` - GraphViz format files
- `*_dependency_graph_adjacency.txt` - Text-based adjacency lists

Output directory structure:
```
output/
├── traces/                         # Raw trace data
│   ├── collected_traces_20250713_143022.json
│   ├── anomalous_traces_20250713_143022.json
│   └── traces.json                 # Sample data
└── analysis/                       # Analysis results
    ├── traces_dependency_graph.png # Visualizations
    ├── my_analysis.json            # Graph exports
    ├── my_analysis.dot             # GraphViz files
    └── my_analysis_adjacency.txt   # Text summaries
```

## Example Usage

Monitor a frontend service every hour:
```bash
export SERVICE_NAME=frontend-service
export POLL_INTERVAL=3600
export LATENCY_PERCENTILE=95
python trace_collector.py
```

## Trace Format

The service works with OpenTelemetry-compatible traces from Jaeger. Each trace contains:
- Trace ID and span information
- Service names and operation details
- Timing data (start time, duration)
- Tags and metadata
