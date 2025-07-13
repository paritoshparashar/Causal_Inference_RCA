import requests
import logging

from .config import Config

logger = logging.getLogger("jaeger-client")

class JaegerClient:
    def __init__(self, config: Config):
        self.config = config
        
    def fetch_traces(self, start_time: int, end_time: int, service_name: str = None) -> list:
        """Retrieve traces from Jaeger within time range"""
        # Use provided service_name, fall back to config, or use default
        service = service_name or self.config.SERVICE_NAME or "frontend-service"
        
        params = {
            "service": service,
            "start": start_time,
            "end": end_time,
            "limit": self.config.MAX_TRACES
        }
        
        # Debug: Log the request details
        url = f"{self.config.jaeger_endpoint}/api/traces"
        logger.info(f"Fetching traces from: {url}")
        logger.debug(f"Request params: {params}")
        
        try:
            response = requests.get(
                url,
                params=params,
                timeout=10
            )
            
            # Debug: Log response details
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            
            json_data = response.json()
            logger.debug(f"Response JSON keys: {list(json_data.keys()) if json_data else 'None'}")
            
            traces = json_data.get("data", [])
            logger.info(f"Successfully fetched {len(traces)} traces from Jaeger")
            
            return traces
            
        except requests.RequestException as e:
            logger.error(f"Jaeger API error: {e}")
            logger.error(f"Failed request URL: {url}")
            logger.error(f"Failed request params: {params}")
            return []