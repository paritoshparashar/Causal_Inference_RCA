"""
Shared utility functions for the causal inference pipeline.
"""
import logging

logger = logging.getLogger(__name__)

def extract_service_name(span: dict, processes: dict) -> str:
    """
    Extract a clean, consistent service name from a span.

    This function prioritizes the service name from the 'processes'
    section of a trace, which is generally more reliable. It falls
    back to parsing the operation name if needed.

    Args:
        span: The span object from the trace.
        processes: The processes dictionary from the trace.

    Returns:
        A cleaned-up service name.
    """
    # 1. Prioritize service name from the 'processes' section
    process_id = span.get('processID')
    if process_id and process_id in processes:
        service_name = processes[process_id].get('serviceName', 'unknown')
        if service_name != 'unknown':
            # Return service name as-is without any cleaning
            return service_name

    # 2. Fallback to parsing the operationName
    operation_name = span.get('operationName', '')
    if not operation_name:
        return 'unknown'

    # Generic parsing for common patterns (e.g., gRPC)
    # Assumes patterns like 'package.Service/Method'
    if '/' in operation_name:
        return operation_name.split('/')[0].lower()

    # Pattern: ServiceNameService_Operation
    if 'Service_' in operation_name:
        return operation_name.split('Service_')[0].lower()
        
    # Pattern: ServiceNameServer_Operation or ServiceNameClient_Operation
    if 'Server_' in operation_name:
        return operation_name.split('Server_')[0].lower().replace('service', '')
    if 'Client_' in operation_name:
        return operation_name.split('Client_')[0].lower().replace('service', '')

    # 3. If no other pattern matches, return a cleaned-up operation name
    return operation_name.lower()
