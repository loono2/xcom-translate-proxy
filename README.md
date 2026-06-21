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
  "target": "zh-CN"
}
```

```json
{
  "text": "你好世界",
  "source": "auto",
  "target": "zh-CN",
  "provider": "google_gtx",
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
```
