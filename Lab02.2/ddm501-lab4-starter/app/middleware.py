"""
Middleware for collecting HTTP metrics.

This module implements a Starlette middleware that automatically
records metrics for all HTTP requests.

TODO: Complete the middleware implementation below.

The middleware should:
1. Record the start time before processing the request
2. Call the next handler to process the request
3. Calculate the duration after the response is ready
4. Record metrics (request count and latency)
"""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.metrics import REQUEST_COUNT, REQUEST_LATENCY


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect HTTP request metrics.
    
    This middleware automatically records:
    - Total request count (by method, endpoint, status)
    - Request latency (by method, endpoint)
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process the request and record metrics.
        
        Args:
            request: The incoming HTTP request
            call_next: The next handler in the chain
            
        Returns:
            The HTTP response
        """
        # TODO 1: Record the start time
        # start_time = time.time()
        
        # TODO 2: Call the next handler to get the response
        # response = await call_next(request)
        
        # TODO 3: Calculate the duration
        # duration = time.time() - start_time
        
        # TODO 4: Extract request information
        # endpoint = request.url.path
        # method = request.method
        # status = response.status_code
        
        # TODO 5: Record the request count metric
        # Only record if REQUEST_COUNT is defined
        # if REQUEST_COUNT is not None:
        #     REQUEST_COUNT.labels(
        #         method=method,
        #         endpoint=endpoint,
        #         status=status
        #     ).inc()
        
        # TODO 6: Record the request latency metric
        # Only record if REQUEST_LATENCY is defined
        # if REQUEST_LATENCY is not None:
        #     REQUEST_LATENCY.labels(
        #         method=method,
        #         endpoint=endpoint
        #     ).observe(duration)
        
        # TODO 7: Return the response
        # return response
        
        # =================================================================
        # Placeholder implementation - replace with your code above
        # =================================================================
        response = await call_next(request)
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log request details (for debugging).
    
    This is an optional middleware that logs request information.
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Log request details and process."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {response.status_code} - {duration:.3f}s"
        )
        
        return response


# =============================================================================
# Helper function to check if metrics are properly configured
# =============================================================================

def check_metrics_middleware_ready() -> dict:
    """
    Check if the metrics middleware is properly configured.
    
    Returns:
        Dictionary with status of each metric
    """
    return {
        'request_count_ready': REQUEST_COUNT is not None,
        'request_latency_ready': REQUEST_LATENCY is not None,
        'middleware_functional': True,
    }
