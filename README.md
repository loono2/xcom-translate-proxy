# Xcom Translate Proxy

Small FastAPI service for Xcom translation requests.

## Endpoints

```http
GET /health
```

```http
POST /v1/translate
Authorization: Bearer <TRANSLATE_PROXY_TOKEN>
Content-Type: application/json
```

```json
{
  "text": "Hello world",
  "source": "auto",
  "target": "zh-CN",
  "provider": "openrouter_deepseek_v4_flash"
}
```

Supported providers:

- `google_gtx`
- `openrouter_deepseek_v4_flash`

```json
{
  "text": "你好世界",
  "source": "auto",
  "target": "zh-CN",
  "provider": "openrouter_deepseek_v4_flash",
  "cached": false
}
```

## Render

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Required environment variable:

```bash
TRANSLATE_PROXY_TOKEN=<secret>
OPENROUTER_API_KEY=<secret>
```

Optional environment variable:

```bash
OPENROUTER_MODEL=deepseek/deepseek-v4-flash
```
