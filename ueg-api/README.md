# UEG REST API

Universal Edge Gateway — intent classifier REST API.
No API keys needed. Model is public on HuggingFace.

## Deploy on Render (recommended)

1. Push this folder to a GitHub repo
2. Go to render.com → New → Web Service
3. Connect your GitHub repo
4. Render detects render.yaml automatically
5. Click Deploy — done

Your API will be live at `https://ueg-api.onrender.com`

## Run locally

```bash
pip install -r requirements.txt
python main.py
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
