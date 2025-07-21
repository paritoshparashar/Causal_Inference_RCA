# Causal Inference RCA System - Clean Project Structure

## Core Files

### Primary Scripts
- **`main.py`** - Main entry point with 4 operation modes:
  - Training from Jaeger/file
  - RCA analysis from Jaeger/file
- **`generate_synthetic_traces.py`** - Generate synthetic OTEL traces with controlled anomalies
- **`visualize_attributions.py`** - Visualize RCA attribution scores

### Configuration
- **`requirements.txt`** - Python dependencies
- **`.env.example`** - Environment variable template
- **`README.md`** - Project documentation

### Source Code (`src/`)
- **`trace_collection/`** - Jaeger client, anomaly detection, configuration
- **`dependency_analysis/`** - Service dependency analysis and graph building
- **`causal_inference/`** - Root cause analysis using DoWhy/GCM
- **`synthetic_traces/`** - Synthetic trace generation system

### Development/Testing
- **`test_causal_setup.py`** - Test causal inference setup
- **`test_synthetic_generation.py`** - Test synthetic trace generation

### Output Structure (`output/`)
- **`traces/`** - Generated/collected trace files
- **`analysis/`** - Dependency analysis results (JSON, DOT, PNG)
- **`models/`** - Trained causal models (PKL files)
- **`rca/`** - Root cause analysis results

## Usage Examples

```bash
# Generate synthetic traces
python generate_synthetic_traces.py --num-traces 100 --slow-service inventory_svc

# Train causal model
python main.py --train-file output/traces/synthetic_traces_inventory_svc_100.json

# Perform RCA
python main.py --rca-file output/traces/synthetic_traces_inventory_svc_100.json \
    --model-path output/models/causal_model_*.pkl --target-service inventory_svc

# Visualize results
python visualize_attributions.py -f output/rca/rca_results_*.json -t single_outlier
```

## Key Features
- ✅ Clean file structure without redundant scripts
- ✅ Comprehensive main.py with all operation modes
- ✅ No automatic file cleanup (manual control)
- ✅ Proper error handling and logging
- ✅ Modular source code architecture
- ✅ Integrated visualization tools
