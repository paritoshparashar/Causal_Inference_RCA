#!/usr/bin/env python3
"""
API Server for Causal Inference RCA System
Provides REST endpoints for training models and performing RCA analysis.
"""

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
import subprocess
import tempfile
from pathlib import Path
import logging
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/app/output'
ALLOWED_EXTENSIONS = {'json'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Train a causal model using uploaded trace file.
    
    Expected form data:
    - traces_file: JSON file containing traces
    
    Returns:
    - model_path: Path to trained model
    - dependency_file: Path to dependency analysis
    """
    try:
        # Check if file is provided
        if 'traces_file' not in request.files:
            return jsonify({'error': 'No traces file provided'}), 400
        
        file = request.files['traces_file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. JSON required.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Run training
        cmd = [
            'python', '/app/main.py',
            '--train-file', filepath
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return jsonify({
                'error': 'Training failed',
                'stderr': result.stderr,
                'stdout': result.stdout
            }), 500
        
        # Find generated files
        models_dir = Path(OUTPUT_FOLDER) / 'models'
        analysis_dir = Path(OUTPUT_FOLDER) / 'analysis'
        
        model_files = list(models_dir.glob('causal_model_*.pkl')) if models_dir.exists() else []
        dependency_files = list(analysis_dir.glob('dependency_analysis_*.json')) if analysis_dir.exists() else []
        
        return jsonify({
            'status': 'success',
            'model_path': str(model_files[-1]) if model_files else None,
            'dependency_file': str(dependency_files[-1]) if dependency_files else None,
            'stdout': result.stdout
        })
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup uploaded file
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)

@app.route('/api/rca', methods=['POST'])
def perform_rca():
    """
    Perform RCA analysis using uploaded trace file and existing model.
    
    Expected form data:
    - traces_file: JSON file containing traces
    - target_service: Service to analyze (optional)
    
    Returns:
    - rca_results: Path to RCA results file
    """
    try:
        # Check if file is provided
        if 'traces_file' not in request.files:
            return jsonify({'error': 'No traces file provided'}), 400
        
        file = request.files['traces_file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. JSON required.'}), 400
        
        # Get target service (optional)
        target_service = request.form.get('target_service', 'frontend')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Find latest model
        models_dir = Path(OUTPUT_FOLDER) / 'models'
        model_files = list(models_dir.glob('causal_model_*.pkl')) if models_dir.exists() else []
        
        if not model_files:
            return jsonify({'error': 'No trained model found. Please train a model first.'}), 400
        
        latest_model = str(sorted(model_files, key=os.path.getmtime)[-1])
        
        # Run RCA analysis
        cmd = [
            'python', '/app/main.py',
            '--rca-file', filepath,
            '--model-path', latest_model,
            '--target-service', target_service
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return jsonify({
                'error': 'RCA analysis failed',
                'stderr': result.stderr,
                'stdout': result.stdout
            }), 500
        
        # Find generated RCA results
        rca_dir = Path(OUTPUT_FOLDER) / 'rca'
        rca_files = list(rca_dir.glob('rca_results_*.json')) if rca_dir.exists() else []
        latest_rca = str(sorted(rca_files, key=os.path.getmtime)[-1]) if rca_files else None
        
        return jsonify({
            'status': 'success',
            'rca_results': latest_rca,
            'target_service': target_service,
            'model_used': latest_model,
            'stdout': result.stdout
        })
        
    except Exception as e:
        logger.error(f"RCA error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup uploaded file
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)

@app.route('/api/generate-synthetic', methods=['POST'])
def generate_synthetic():
    """
    Generate synthetic traces with controlled anomalies.
    
    Expected JSON body:
    - num_traces: Number of traces to generate (default: 100)
    - slow_service: Service to inject slowness (optional)
    
    Returns:
    - traces_file: Path to generated traces
    """
    try:
        data = request.get_json() or {}
        num_traces = data.get('num_traces', 100)
        slow_service = data.get('slow_service')
        
        # Build command
        cmd = [
            'python', '/app/generate_synthetic_traces.py',
            '--num-traces', str(num_traces)
        ]
        
        if slow_service:
            cmd.extend(['--slow-service', slow_service])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return jsonify({
                'error': 'Synthetic trace generation failed',
                'stderr': result.stderr,
                'stdout': result.stdout
            }), 500
        
        # Find generated traces
        traces_dir = Path(OUTPUT_FOLDER) / 'traces'
        trace_files = list(traces_dir.glob('synthetic_traces_*.json')) if traces_dir.exists() else []
        latest_traces = str(sorted(trace_files, key=os.path.getmtime)[-1]) if trace_files else None
        
        return jsonify({
            'status': 'success',
            'traces_file': latest_traces,
            'num_traces': num_traces,
            'slow_service': slow_service,
            'stdout': result.stdout
        })
        
    except Exception as e:
        logger.error(f"Synthetic generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/files/<path:filename>', methods=['GET'])
def download_file(filename):
    """Download generated files."""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status and available files."""
    try:
        output_path = Path(OUTPUT_FOLDER)
        
        status = {
            'models': [],
            'traces': [],
            'rca_results': [],
            'analysis': []
        }
        
        # List available files
        if (output_path / 'models').exists():
            status['models'] = [f.name for f in (output_path / 'models').glob('*.pkl')]
            
        if (output_path / 'traces').exists():
            status['traces'] = [f.name for f in (output_path / 'traces').glob('*.json')]
            
        if (output_path / 'rca').exists():
            status['rca_results'] = [f.name for f in (output_path / 'rca').glob('*.json')]
            
        if (output_path / 'analysis').exists():
            status['analysis'] = [f.name for f in (output_path / 'analysis').glob('*')]
        
        return jsonify({
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'available_files': status
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
