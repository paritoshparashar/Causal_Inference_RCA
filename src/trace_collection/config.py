import os

class Config:
    """Centralized configuration management"""
    JAEGER_QUERY_HOST = os.getenv("JAEGER_QUERY_HOST", "localhost")
    JAEGER_QUERY_PORT = os.getenv("JAEGER_QUERY_PORT", "16686")
    SERVICE_NAME = os.getenv("SERVICE_NAME", "frontend-service")  # Default to our microservice name
    PAST_INTERVAL = int(os.getenv("PAST_INTERVAL", "3600"))  # seconds = 1 hour
    POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "43200"))  # seconds = 1 hour
    LATENCY_PERCENTILE = float(os.getenv("LATENCY_PERCENTILE", "99"))  # 99th percentile
    MAX_TRACES = int(os.getenv("MAX_TRACES", "10000"))
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/traces")  # Output directory for trace files
    AUTO_ANALYZE = os.getenv("AUTO_ANALYZE", "true").lower() == "true"  # Auto-run dependency analysis

    @property
    def jaeger_endpoint(self):
        return f"http://{self.JAEGER_QUERY_HOST}:{self.JAEGER_QUERY_PORT}"