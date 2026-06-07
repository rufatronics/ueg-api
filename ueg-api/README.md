# UEG REST API

Universal Edge Gateway — intent classifier REST API.

## Setup

```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN

pip install -r requirements.txt
python main.py
```

## Docker

```bash
docker build -t ueg-api .
docker run -p 8000:8000 --env-file .env ueg-api
```

## Endpoints

### POST /classify
```json
{
  "text": "Write a Python function to reverse a string",
  "include_probabilities": false
}
```

Response:
```json
{
  "intent_class_id": 13,
  "intent_class_label": "code_task",
  "tier": "5A",
  "routing_action": "route_to_frontier",
  "confidence_intent": 0.9821,
  "resource_class": "hr_global",
  "confidence_resource": 0.9934,
  "language_iso": "en",
  "language_confidence": 0.9812,
  "latency_ms": 1.24,
  "model": "ueg-classifier-v1"
}
```

### POST /classify/batch
```json
{
  "texts": ["Hello!", "Write a SQL query", "What time is it?"],
  "include_probabilities": false
}
```

### GET /health
### GET /info

## Routing actions
| Action | Tier | Meaning |
|--------|------|---------|
| `drop` | 1 | Noise — discard silently |
| `block` | 1 | Adversarial — reject with warning |
| `static_template` | 2 | Respond with pre-written template |
| `device_api` | 3 | Call local device/environment API |
| `cache_lookup` | 4 | Check cache or micro-LLM |
| `route_to_frontier` | 5A/5B | Send to full frontier model |

## Interactive docs
`http://localhost:8000/docs`
