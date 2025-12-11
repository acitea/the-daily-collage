# API Testing Guide

This guide provides examples for testing The Daily Collage REST API endpoints.

## Prerequisites

1. Backend server running on `http://localhost:8000`
2. `curl` command-line tool or Postman/Insomnia
3. `jq` for pretty-printing JSON (optional)

## Starting the Server

```bash
cd backend/server
uv sync
uv run python -m uvicorn main:app --reload
```

## Health Check Endpoints

### Check API Status

```bash
curl http://localhost:8000/
```

Expected response:
```json
{
  "name": "The Daily Collage API",
  "status": "operational",
  "version": "0.1.0"
}
```

### Detailed Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-11T14:30:00.000Z"
}
```

## Visualization Endpoints

### Get Visualization for Sweden

```bash
curl http://localhost:8000/api/visualization?location=sweden | jq
```

Response:
```json
{
  "location": "sweden",
  "generated_at": "2025-12-11T14:30:00.000Z",
  "signal_count": 3,
  "signals": [
    {"name": "weather", "intensity": 65.0},
    {"name": "traffic", "intensity": 45.0},
    {"name": "politics", "intensity": 75.0}
  ],
  "image_url": "/api/visualization/sweden/image",
  "cached": false
}
```

### Get Visualization for Stockholm

```bash
curl http://localhost:8000/api/visualization?location=stockholm | jq
```

### Force Regeneration (Skip Cache)

```bash
curl "http://localhost:8000/api/visualization?location=sweden&force_regenerate=true" | jq
```

### Get Multiple Visualizations

```bash
for location in sweden stockholm gothenburg malmo; do
  echo "=== $location ==="
  curl -s "http://localhost:8000/api/visualization?location=$location" | jq '.signals'
done
```

### Get Visualization Image (Placeholder)

```bash
curl http://localhost:8000/api/visualization/sweden/image | jq
```

## Article Endpoints

### Get All Articles for Sweden

```bash
curl http://localhost:8000/api/articles?location=sweden | jq
```

Response:
```json
{
  "location": "sweden",
  "articles": [
    {
      "title": "Heavy traffic on Stockholm ring road",
      "url": "https://example.com/article1",
      "source": "SVT Nyheter",
      "date": "2025-12-11T14:30:00Z",
      "signal": "traffic"
    },
    {
      "title": "Rainstorm expected in southern Sweden",
      "url": "https://example.com/article2",
      "source": "SMHI",
      "date": "2025-12-11T10:15:00Z",
      "signal": "weather"
    }
  ]
}
```

### Filter Articles by Signal

```bash
# Get only weather-related articles
curl "http://localhost:8000/api/articles?location=sweden&signal=weather" | jq

# Get only traffic-related articles
curl "http://localhost:8000/api/articles?location=stockholm&signal=traffic" | jq
```

### Get Articles for Multiple Signals

```bash
for signal in weather traffic politics; do
  echo "=== $signal ==="
  curl -s "http://localhost:8000/api/articles?location=sweden&signal=$signal" | jq '.articles | length'
done
```

## Metadata Endpoints

### Get Supported Locations

```bash
curl http://localhost:8000/api/supported-locations | jq
```

Response:
```json
{
  "locations": [
    {
      "code": "se",
      "name": "Sweden",
      "type": "country"
    },
    {
      "code": "stockholm",
      "name": "Stockholm",
      "type": "city"
    },
    {
      "code": "gothenburg",
      "name": "Gothenburg",
      "type": "city"
    },
    {
      "code": "malmo",
      "name": "Malmö",
      "type": "city"
    }
  ]
}
```

### Get All Signal Categories

```bash
curl http://localhost:8000/api/signal-categories | jq
```

Response:
```json
{
  "categories": [
    {
      "id": "traffic",
      "name": "Traffic & Transportation",
      "description": "Road congestion, public transit disruptions",
      "icon": "🚗"
    },
    {
      "id": "weather",
      "name": "Weather Events",
      "description": "Storms, heatwaves, snow, flooding",
      "icon": "🌧️"
    },
    ...
  ]
}
```

### Get Just Signal Names

```bash
curl -s http://localhost:8000/api/signal-categories | jq '.categories[].name'
```

Output:
```
"Traffic & Transportation"
"Weather Events"
"Crime & Safety"
"Festivals & Events"
"Politics"
"Sports"
"Accidents & Emergencies"
"Economic"
```

## Monitoring Endpoints

### Get Cache Statistics

```bash
curl http://localhost:8000/api/cache-stats | jq
```

Response:
```json
{
  "cache": {
    "cached_visualizations": 5,
    "cache_size_estimates": 2048576
  }
}
```

### Monitor Cache Growth

```bash
for i in {1..10}; do
  count=$(curl -s http://localhost:8000/api/cache-stats | jq '.cache.cached_visualizations')
  echo "$(date): $count cached visualizations"
  sleep 1
done
```

## Advanced Testing

### Test Cache Behavior

```bash
# First request (cache miss)
time curl -s "http://localhost:8000/api/visualization?location=sweden" > /dev/null

# Second request (cache hit - should be faster)
time curl -s "http://localhost:8000/api/visualization?location=sweden" > /dev/null

# Force regenerate (cache miss again)
time curl -s "http://localhost:8000/api/visualization?location=sweden&force_regenerate=true" > /dev/null
```

### Load Testing with Apache Bench

```bash
# 100 requests, 10 concurrent
ab -n 100 -c 10 "http://localhost:8000/api/visualization?location=sweden"

# Higher concurrency
ab -n 1000 -c 50 "http://localhost:8000/api/visualization?location=sweden"
```

### Load Testing with Locust

Create `locustfile.py`:

```python
from locust import HttpUser, task, between

class DailyCollageUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def get_visualization(self):
        self.client.get("/api/visualization?location=sweden")

    @task(1)
    def get_articles(self):
        self.client.get("/api/articles?location=sweden")

    @task(1)
    def get_metadata(self):
        self.client.get("/api/supported-locations")
```

Run with:
```bash
locust -f locustfile.py -u 100 -r 10 --host=http://localhost:8000
```

### Test All Endpoints

Create `test_all_endpoints.sh`:

```bash
#!/bin/bash

BASE_URL="http://localhost:8000"
FAILED=0
PASSED=0

test_endpoint() {
  local method=$1
  local endpoint=$2
  local expected_status=$3

  echo -n "Testing $method $endpoint ... "
  status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL$endpoint")
  
  if [ "$status" = "$expected_status" ]; then
    echo "✓ ($status)"
    ((PASSED++))
  else
    echo "✗ (got $status, expected $expected_status)"
    ((FAILED++))
  fi
}

# Health endpoints
test_endpoint GET "/" 200
test_endpoint GET "/health" 200

# Visualization endpoints
test_endpoint GET "/api/visualization" 200
test_endpoint GET "/api/visualization?location=stockholm" 200
test_endpoint GET "/api/visualization/sweden/image" 200

# Article endpoints
test_endpoint GET "/api/articles" 200
test_endpoint GET "/api/articles?location=stockholm&signal=weather" 200

# Metadata endpoints
test_endpoint GET "/api/supported-locations" 200
test_endpoint GET "/api/signal-categories" 200
test_endpoint GET "/api/cache-stats" 200

echo ""
echo "Results: $PASSED passed, $FAILED failed"
exit $FAILED
```

Run with:
```bash
chmod +x test_all_endpoints.sh
./test_all_endpoints.sh
```

## Using Postman

### Import API Collection

1. Create new collection "Daily Collage"
2. Add requests:

**Health Check**
- GET http://localhost:8000/
- GET http://localhost:8000/health

**Visualizations**
- GET http://localhost:8000/api/visualization
- GET http://localhost:8000/api/visualization?location=stockholm

**Articles**
- GET http://localhost:8000/api/articles
- GET http://localhost:8000/api/articles?location=sweden&signal=weather

**Metadata**
- GET http://localhost:8000/api/supported-locations
- GET http://localhost:8000/api/signal-categories

**Cache**
- GET http://localhost:8000/api/cache-stats

### Test Environment Variables

Set variables in Postman environment:

```json
{
  "base_url": "http://localhost:8000",
  "location": "sweden",
  "signal": "weather"
}
```

Use in requests:
```
{{base_url}}/api/visualization?location={{location}}
```

## Error Testing

### Invalid Location

```bash
curl "http://localhost:8000/api/articles?location=invalid" | jq
```

### Test Error Responses

```bash
# This should still work (404 might not be implemented for invalid locations yet)
curl -v "http://localhost:8000/api/visualization?location=atlantis" | jq
```

## Performance Benchmarking

### Response Time Measurement

```bash
# Measure single request time
curl -w "\nTotal time: %{time_total}s\n" -s -o /dev/null http://localhost:8000/api/visualization

# Measure multiple sequential requests
for i in {1..5}; do
  curl -w "Request $i: %{time_total}s\n" -s -o /dev/null "http://localhost:8000/api/visualization?location=sweden&force_regenerate=true"
done
```

### Compare Cached vs Non-Cached

```bash
echo "=== Cached Response ==="
time curl -s "http://localhost:8000/api/visualization?location=sweden" > /dev/null

echo ""
echo "=== Non-Cached Response ==="
time curl -s "http://localhost:8000/api/visualization?location=sweden&force_regenerate=true" > /dev/null
```

## Continuous Monitoring

Watch the API health continuously:

```bash
watch -n 5 'curl -s http://localhost:8000/api/cache-stats | jq'
```

## Integration with Frontend

Test the frontend's API connectivity:

```bash
# Serve frontend on port 8080 using Python
python -m http.server 8080 --directory frontend

# Open browser and test API calls work
# Check browser console for any CORS or connection errors
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: API Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: astral-sh/setup-uv@v2
      - run: |
          cd backend/server
          uv sync
          uv run python -m uvicorn main:app &
          sleep 2
          bash ../../tests/test_all_endpoints.sh
```

## Common Issues

### Connection Refused

```bash
# Verify server is running
netstat -an | grep 8000

# Or check with curl
curl http://localhost:8000/ 2>&1
```

### JSON Parsing Errors

```bash
# Validate JSON response
curl -s http://localhost:8000/api/visualization | jq empty
```

### CORS Issues (Frontend)

Server needs CORS middleware:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## More Resources

- OpenAPI/Swagger docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Backend README: `backend/server/README.md`
- Main documentation: `README.md`
