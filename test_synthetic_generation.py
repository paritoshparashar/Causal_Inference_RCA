#!/usr/bin/env python3
"""
Test script for synthetic trace generation.

Quick validation that the trace generator works correctly.
"""

import sys
from pathlib import Path

# Add src to Python path  
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.synthetic_traces import generate_traces


def test_trace_generation():
    """Test basic trace generation functionality."""
    print("Testing Synthetic Trace Generation...")
    print("=" * 50)
    
    try:
        # Test with a small number of traces
        output_file = generate_traces(
            num_traces=50,
            slow_service='payment_svc',
            output_file='output/traces/test_synthetic_traces.json'
        )
        
        # Verify file was created and has content
        with open(output_file, 'r') as f:
            import json
            traces = json.load(f)
        
        print(f"\\n✓ Generated {len(traces)} traces successfully")
        print(f"✓ Output file: {output_file}")
        
        # Check first trace structure
        if traces:
            first_trace = traces[0]
            print(f"✓ First trace has {len(first_trace['spans'])} spans")
            print(f"✓ First trace has {len(first_trace['processes'])} processes")
            
            # Show service names
            services = set()
            for span in first_trace['spans']:
                for tag in span['tags']:
                    if tag['key'] == 'service.name':
                        services.add(tag['value'])
            
            print(f"✓ Services in trace: {sorted(services)}")
        
        print("\\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_trace_generation()
    sys.exit(0 if success else 1)
