#!/bin/bash
# Production deployment script for Causal RCA System

set -e

echo "Deploying Causal Inference RCA System to Production"
echo "====================================================="

# Configuration
IMAGE_NAME="causal-rca"
IMAGE_TAG="${1:-latest}"
REGISTRY="${REGISTRY:-your-registry.com}"

echo "Deployment Configuration:"
echo "   Image: $REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
echo "   Environment: ${ENVIRONMENT:-production}"
echo ""

# Build and tag image
echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME:$IMAGE_TAG .
docker tag $IMAGE_NAME:$IMAGE_TAG $REGISTRY/$IMAGE_NAME:$IMAGE_TAG

echo "Image built and tagged"

# Push to registry (uncomment for actual deployment)
# echo "Pushing to registry..."
# docker push $REGISTRY/$IMAGE_NAME:$IMAGE_TAG

# Deploy with docker-compose
echo "Deploying services..."
export IMAGE_TAG
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

echo "Waiting for services to be ready..."
sleep 10

# Health check
echo "Performing health check..."
if curl -f -s http://localhost:8000/health > /dev/null; then
    echo "API server is healthy"
else
    echo "API server health check failed"
    docker-compose logs rca-api
    exit 1
fi

echo "Deployment completed successfully!"
echo ""
echo "Service URLs:"
echo "   API Server: http://localhost:8000"
echo "   Jaeger UI: http://localhost:16686"
echo "   Redis: redis://localhost:6379"
echo ""
echo "API Documentation:"
echo "   Health: GET http://localhost:8000/health"
echo "   Status: GET http://localhost:8000/api/status"
echo "   Train: POST http://localhost:8000/api/train"
echo "   RCA: POST http://localhost:8000/api/rca"
