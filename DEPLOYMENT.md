# RealtimeSTT Multi-Backend Deployment Guide

## Docker Compose Deployment

### Quick Start

#### GPU Deployment (Recommended for Parakeet backend)
```bash
# Build and start with GPU support
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

#### CPU-Only Deployment
```bash
# Edit docker-compose.yml: change 'target: gpu' to 'target: cpu'
# Remove the 'deploy' section for GPU resources
docker-compose up --build
```

### Backend Selection

#### Default (faster_whisper)
```bash
# Uses the default command in docker-compose.yml
docker-compose up
```

#### Parakeet Backend
```bash
# Override the default command
docker-compose run --rm -p 8011:8011 -p 8012:8012 realtimestt-server python3 stt_server.py --backend parakeet
```

#### Custom Configuration
```bash
# Run with specific model and settings
docker-compose run --rm -p 8011:8011 -p 8012:8012 realtimestt-server python3 stt_server.py --backend faster_whisper --model base --language en
```

### Environment Variables

You can customize the deployment by setting environment variables:

```yaml
# In docker-compose.yml, add to environment section:
environment:
  - KMP_DUPLICATE_LIB_OK=TRUE
  - PYTHONPATH=/app
  - CUDA_VISIBLE_DEVICES=0  # Specify GPU device
  - WHISPER_CACHE_DIR=/app/models  # Custom model cache
```

### Volume Mounts

#### Development Mode
Uncomment the volume mounts in docker-compose.yml for live code changes:
```yaml
volumes:
  - ./RealtimeSTT:/app/RealtimeSTT
  - ./RealtimeSTT_server:/app/RealtimeSTT_server
  - cache:/root/.cache
  - models:/app/models
```

#### Production Mode
Use only the cache and models volumes (default configuration).

### Port Configuration

The service exposes two ports:
- **8011**: WebSocket control connection
- **8012**: WebSocket data connection

To change ports, modify the docker-compose.yml:
```yaml
ports:
  - "9011:8011"  # Map to different host port
  - "9012:8012"
```

### Health Checks

The service includes health checks to monitor server status:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8011", "||", "exit", "1"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Direct Docker Commands

### Build Images

#### GPU Image
```bash
docker build --target gpu -t realtimestt-gpu .
```

#### CPU Image
```bash
docker build --target cpu -t realtimestt-cpu .
```

### Run Containers

#### GPU Container with faster_whisper
```bash
docker run -d \
  --name realtimestt-server \
  --gpus all \
  -p 8011:8011 -p 8012:8012 \
  -e KMP_DUPLICATE_LIB_OK=TRUE \
  realtimestt-gpu
```

#### GPU Container with Parakeet
```bash
docker run -d \
  --name realtimestt-parakeet \
  --gpus all \
  -p 8011:8011 -p 8012:8012 \
  -e KMP_DUPLICATE_LIB_OK=TRUE \
  realtimestt-gpu \
  python3 stt_server.py --backend parakeet
```

#### CPU Container
```bash
docker run -d \
  --name realtimestt-cpu \
  -p 8011:8011 -p 8012:8012 \
  -e KMP_DUPLICATE_LIB_OK=TRUE \
  realtimestt-cpu
```

## Client Connection

### WebSocket Endpoints
- **Control**: `ws://localhost:8011`
- **Data**: `ws://localhost:8012`

### Test Connection
```bash
# Use the CLI client
python stt_cli_client.py --control "ws://localhost:8011" --data "ws://localhost:8012"
```

## Troubleshooting

### GPU Issues
```bash
# Check GPU availability
docker run --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi

# Check container GPU access
docker exec -it realtimestt-server nvidia-smi
```

### Memory Issues
```bash
# Monitor container resources
docker stats realtimestt-server

# Increase shared memory if needed
docker run --shm-size=2g ...
```

### Model Download Issues
```bash
# Check model cache volume
docker volume inspect stt3_models

# Clear model cache if needed
docker volume rm stt3_models
```

### Logs
```bash
# View container logs
docker-compose logs realtimestt-server

# Follow logs in real-time
docker-compose logs -f realtimestt-server
```

## Production Considerations

### Resource Limits
```yaml
# Add to docker-compose.yml service
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4'
    reservations:
      memory: 4G
      cpus: '2'
      devices:
      - capabilities: ["gpu"]
```

### Restart Policy
```yaml
# Add to docker-compose.yml service
restart: unless-stopped
```

### Network Configuration
```yaml
# Custom network
networks:
  realtimestt-net:
    driver: bridge

services:
  realtimestt-server:
    networks:
      - realtimestt-net
```

### SSL/TLS (for production)
Consider using a reverse proxy like nginx or traefik for SSL termination:
```yaml
# Add nginx service for SSL
nginx:
  image: nginx:latest
  ports:
    - "443:443"
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
    - ./ssl:/etc/nginx/ssl
  depends_on:
    - realtimestt-server
```

## Scaling

### Multiple Instances
```bash
# Scale to multiple instances
docker-compose up --scale realtimestt-server=3
```

### Load Balancing
Use nginx or HAProxy to distribute load across multiple instances.

## Monitoring

### Health Monitoring
The built-in health check monitors server availability. For production, consider:
- Prometheus metrics
- Grafana dashboards
- Log aggregation (ELK stack)

### Performance Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor container performance
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
```
