import asyncio
import hashlib
import os
import random
import re
import time
from collections import OrderedDict, defaultdict, deque
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field


GOOGLE_TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
DEFAULT_MAX_CHARS_PER_SEGMENT = 4200
DEFAULT_CACHE_TTL_SECONDS = 60 * 60 * 24
DEFAULT_CACHE_MAX_ITEMS = 2048


app = FastAPI(title="Xcom Translate Proxy", version="1.0.0")


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = Field(default="auto", min_length=2, max_length=16)
    target: str = Field(default="zh-CN", min_length=2, max_length=16)


class TranslateResponse(BaseModel):
    text: str
    source: str
    target: str
    provider: str = "google_gtx"
    cached: bool


class CacheEntry(BaseModel):
    value: str
    expires_at: float


class TranslationCache:
    def __init__(self, max_items: int, ttl_seconds: int) -> None:
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            entry = self._entries.get(key)
            now = time.time()
            if entry is None:
                return None

            if entry.expires_at <= now:
                self._entries.pop(key, None)
                return None

            self._entries.move_to_end(key)
            return entry.value

    async def set(self, key: str, value: str) -> None:
        async with self._lock:
            self._entries[key] = CacheEntry(
                value=value,
                expires_at=time.time() + self.ttl_seconds,
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_items:
                self._entries.popitem(last=False)


class RateLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self.limit_per_minute = limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> None:
        if self.limit_per_minute <= 0:
            return

        now = time.monotonic()
        window_start = now - 60
        async with self._lock:
            hits = self._hits[key]
            while hits and hits[0] < window_start:
                hits.popleft()

            if len(hits) >= self.limit_per_minute:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                )

            hits.append(now)


def int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default


def configured_token() -> str:
    return os.getenv("TRANSLATE_PROXY_TOKEN", "").strip()


def max_text_chars() -> int:
    return int_env("MAX_TEXT_CHARS", 20_000)


def max_chars_per_segment() -> int:
    return int_env("MAX_CHARS_PER_SEGMENT", DEFAULT_MAX_CHARS_PER_SEGMENT)


cache = TranslationCache(
    max_items=int_env("CACHE_MAX_ITEMS", DEFAULT_CACHE_MAX_ITEMS),
    ttl_seconds=int_env("CACHE_TTL_SECONDS", DEFAULT_CACHE_TTL_SECONDS),
)
rate_limiter = RateLimiter(limit_per_minute=int_env("RATE_LIMIT_PER_MINUTE", 60))


async def require_authorization(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    token = configured_token()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server token is not configured",
        )

    expected = f"Bearer {token}"
    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization token",
        )

    client_host = request.client.host if request.client else "unknown"
    await rate_limiter.check(f"{token}:{client_host}")


def cache_key(source: str, target: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{source}:{target}:{digest}"


def split_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]

    units = split_into_units(text, limit)
    segments: list[str] = []
    current = ""

    for unit in units:
        if not current:
            current = unit
            continue

        if len(current) + len(unit) <= limit:
            current += unit
            continue

        segments.append(current)
        current = unit

    if current:
        segments.append(current)

    return segments


def split_into_units(text: str, limit: int) -> list[str]:
    raw_units = re.split(r"(?<=[.!?。！？；;])(\s+|$)|(\n+)", text)
    units = [unit for unit in raw_units if unit]
    if not units:
        units = [text]

    normalized_units: list[str] = []
    for unit in units:
        if len(unit) <= limit:
            normalized_units.append(unit)
            continue

        for start in range(0, len(unit), limit):
            normalized_units.append(unit[start : start + limit])

    return normalized_units


async def google_translate(text: str, source: str, target: str) -> str:
    segments = split_text(text, max_chars_per_segment())
    translated_segments: list[str] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
        for segment in segments:
            translated_segments.append(
                await google_translate_segment(client, segment, source, target)
            )

    return "".join(translated_segments)


async def google_translate_segment(
    client: httpx.AsyncClient,
    text: str,
    source: str,
    target: str,
) -> str:
    params = {
        "client": "gtx",
        "sl": source,
        "tl": target,
        "dt": "t",
        "q": text,
    }

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = await client.get(GOOGLE_TRANSLATE_URL, params=params)
            if response.status_code == 429:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Upstream translation rate limit exceeded",
                )

            if response.status_code >= 500:
                raise httpx.HTTPStatusError(
                    "Upstream server error",
                    request=response.request,
                    response=response,
                )

            if response.status_code >= 400:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Upstream translation failed with HTTP {response.status_code}",
                )

            return parse_google_response(response.json())
        except HTTPException:
            raise
        except (httpx.HTTPError, ValueError, TypeError, IndexError) as error:
            last_error = error
            if attempt == 3:
                break

            backoff = (0.4 * (2**attempt)) + random.uniform(0.0, 0.25)
            await asyncio.sleep(backoff)

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"Translation upstream unavailable: {last_error}",
    )


def parse_google_response(data: Any) -> str:
    translations = data[0]
    if not isinstance(translations, list):
        raise ValueError("Unexpected upstream response")

    return "".join(
        str(segment[0])
        for segment in translations
        if isinstance(segment, list) and segment and segment[0] is not None
    )


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/translate", response_model=TranslateResponse)
async def translate(
    payload: TranslateRequest,
    _: None = Depends(require_authorization),
) -> TranslateResponse:
    if len(payload.text) > max_text_chars():
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Text exceeds {max_text_chars()} characters",
        )

    key = cache_key(payload.source, payload.target, payload.text)
    cached_value = await cache.get(key)
    if cached_value is not None:
        return TranslateResponse(
            text=cached_value,
            source=payload.source,
            target=payload.target,
            cached=True,
        )

    translated_text = await google_translate(
        payload.text,
        source=payload.source,
        target=payload.target,
    )
    await cache.set(key, translated_text)

    return TranslateResponse(
        text=translated_text,
        source=payload.source,
        target=payload.target,
        cached=False,
    )
