#!/usr/bin/env python3
"""
Causal Inference RCA System - Usage Guide

This guide shows the new clean command-line interface for the RCA system.
"""

def print_usage_guide():
    print("""
🎯 CAUSAL INFERENCE RCA SYSTEM - CLEAN USAGE GUIDE
==================================================

TRAINING MODES (Build and save causal models):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ TRAIN WITH JAEGER BACKEND:
   python main.py --train-jaeger --jaeger-url http://localhost:16686
   
   ✅ What it does:
   - Connects to Jaeger and fetches fresh traces
   - Detects anomalous traces  
   - Builds dependency graph
   - Trains causal inference model
   - Saves model for later RCA use
   
   📁 Outputs:
   - output/traces/collected_traces_*.json
   - output/traces/anomalous_traces_*.json  
   - output/analysis/dependency_analysis_*.json
   - output/models/causal_model_*.pkl

2️⃣ TRAIN WITH LOCAL TRACE FILE:
   python main.py --train-file output/traces/traces.json
   
   ✅ What it does:
   - Loads existing trace file (109MB traces.json)
   - Detects anomalous traces
   - Builds dependency graph  
   - Trains causal inference model
   - Saves model for later RCA use
   
   📁 Outputs:
   - output/traces/anomalous_traces_*.json
   - output/analysis/dependency_analysis_*.json
   - output/models/causal_model_*.pkl

RCA ANALYSIS MODES (Use trained models for root cause analysis):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3️⃣ RCA WITH JAEGER TRACES:
   python main.py --rca-jaeger \\
       --jaeger-url http://localhost:16686 \\
       --model-path output/models/causal_model_*.pkl \\
       --target-service unknown_service:frontend_proc
   
   ✅ What it does:
   - Connects to Jaeger and fetches fresh traces
   - Detects anomalous traces
   - Uses existing trained model (no retraining)
   - Performs RCA on target service
   - Provides root cause attributions
   
   📁 Outputs:
   - output/traces/collected_traces_*.json
   - output/traces/anomalous_traces_*.json
   - output/rca/rca_results_*.json

4️⃣ RCA WITH LOCAL TRACE FILE:
   python main.py --rca-file output/traces/traces.json \\
       --model-path output/models/causal_model_*.pkl \\
       --target-service unknown_service:frontend_proc
   
   ✅ What it does:
   - Loads existing trace file
   - Detects anomalous traces
   - Uses existing trained model (no retraining)
   - Performs RCA on target service
   - Provides root cause attributions
   
   📁 Outputs:
   - output/traces/anomalous_traces_*.json
   - output/rca/rca_results_*.json

AVAILABLE TARGET SERVICES (from your real traces):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Based on your trace analysis, use these service names:

✅ unknown_service:frontend_proc      (Entry point service)
✅ unknown_service:luggage_proc       (Luggage management)  
✅ unknown_service:search_proc        (Search functionality)
✅ unknown_service:reservation_proc   (Booking system)
✅ unknown_service:userprofile_proc   (User management)
✅ unknown_service:review_proc        (Review system)

WORKFLOW EXAMPLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔄 TYPICAL WORKFLOW:

Step 1 - Train a model (one time setup):
python main.py --train-file output/traces/traces.json

Step 2 - Perform RCA analysis (multiple times as needed):
python main.py --rca-file output/traces/traces.json \\
    --model-path output/models/causal_model_*.pkl \\
    --target-service unknown_service:luggage_proc

Step 3 - Analyze different services:
python main.py --rca-file output/traces/traces.json \\
    --model-path output/models/causal_model_*.pkl \\
    --target-service unknown_service:search_proc

🚀 PRODUCTION WORKFLOW:

Step 1 - Train with production data:
python main.py --train-jaeger --jaeger-url http://prod-jaeger:16686

Step 2 - Regular RCA analysis:  
python main.py --rca-jaeger \\
    --jaeger-url http://prod-jaeger:16686 \\
    --model-path output/models/causal_model_*.pkl \\
    --target-service unknown_service:frontend_proc

LEGACY MODE (for backwards compatibility):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python main.py --continuous --interval 600    # Old continuous mode

ERROR HANDLING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Missing required arguments:
   - RCA modes require --model-path and --target-service
   - Jaeger modes require --jaeger-url

❌ File not found errors:  
   - Check trace file paths exist
   - Check model file paths exist
   - Verify Jaeger URL is accessible

❌ Service not found:
   - Use service names from the available list above
   - Check if service exists in your trace data

HELP:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python main.py --help    # Show all available options
    """)

if __name__ == "__main__":
    print_usage_guide()
