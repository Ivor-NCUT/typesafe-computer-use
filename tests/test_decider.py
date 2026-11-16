"""The Perplexity decision client, with the HTTP stack mocked. No network and no real key."""

from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from typesafe_computer_use.config import decider_model
from typesafe_computer_use.decider import (
    MAX_IMAGES,
    MAX_TILES,
    Choice,
    DecisionError,
    Noul,
    PerplexityDecider,
    fit_size,
    fits_tiles,
    tile_count,
    with_screenshot,
)

KEY = "not-a-real-key"


def _client(handler) -> PerplexityDecider:
    from perplexity import Perplexity

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return PerplexityDecider(client=Perplexity(api_key=KEY, http_client=http, max_retries=0))


def _png(url: str) -> Image.Image:
    encoded = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def test_sizes_that_already_fit_are_unchanged_and_1600x1310_is_shrunk():
    assert fit_size(1440, 1440) == (1440, 1440)
    assert fit_size(2048, 1024) == (2048, 1024)
    assert tile_count(1600, 1310) > MAX_TILES
    width, height = fit_size(1600, 1310)
    assert fits_tiles(width, height)
    assert width <= 1600 and height <= 1310 and (width, height) != (1600, 1310)
    assert fits_tiles(*fit_size(3024, 1964))
    assert fit_size(1, 1) == (1, 1)


def test_a_ninth_image_is_not_added():
    part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,aaaa"}}
    state = {str(i): part for i in range(MAX_IMAGES)}
    image = Image.new("RGB", (8, 8), "red")
    assert with_screenshot(state, image) is state
    assert with_screenshot({"goal": "go"}, None) == {"goal": "go"}


def test_system_one_posts_bearer_auth_choices_and_a_resized_png(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_DECIDER_MODEL", raising=False)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = {key.lower(): value for key, value in request.headers.items()}
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            headers={"x-request-id": "req_test"},
            json={
                "model": "pplx-decider-latest",
                "answers": {
                    "kind": {
                        "type": "choice",
                        "choice": "done",
                        "confidence": 0.8,
                        "probabilities": {"done": 0.9, "wait": 0.1},
                    },
                    "ok": {"type": "noul", "noul": 0.2},
                },
                "usage": {"input_tokens": 12, "output_tokens": 2},
            },
        )

    image = Image.new("RGB", (1600, 1310), "blue")
    reply = _client(handler).system_one(
        state={"goal": "open the report"},
        questions={
            "kind": Choice("Which action?", {"done": "The goal is achieved", "wait": "The page is loading"}),
            "ok": Noul("Is the goal achieved?", {"true": "yes", "false": "no"}),
        },
        image=image,
    )

    assert seen["url"] == "https://api.perplexity.ai/v1/decisions"
    assert seen["headers"]["authorization"] == f"Bearer {KEY}"
    assert "x-api-key" not in seen["headers"]
    body = seen["body"]
    assert set(body) == {"model", "state", "questions"}
    assert body["model"] == "pplx-decider-latest"
    assert "mode" not in body
    assert body["questions"]["kind"] == {
        "type": "choice",
        "instructions": "Which action?",
        "criteria": {"done": "The goal is achieved", "wait": "The page is loading"},
    }
    assert body["questions"]["ok"]["type"] == "noul"
    assert body["questions"]["ok"]["criteria"] == {"true": "yes", "false": "no"}
    assert list(body["state"])[:2] == ["screenshot", "goal"]
    assert body["state"]["goal"] == "open the report"
    url = body["state"]["screenshot"]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert not url.startswith("http")
    assert fits_tiles(*_png(url).size)
    assert _png(url).size != (1600, 1310)
    assert reply.answers["kind"].choice == "done"
    assert reply.answers["kind"].probabilities["wait"] == 0.1
    assert reply.answers["ok"].noul == 0.2
    assert reply.model == "pplx-decider-latest"
    assert reply.usage.input_tokens == 12
    assert reply.request_id == "req_test"
    assert KEY not in request_text(body)


def test_model_env_and_per_call_override(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["model"] = json.loads(request.content.decode())["model"]
        return httpx.Response(
            200,
            json={
                "model": seen["model"],
                "answers": {"kind": {"type": "choice", "choice": "done", "confidence": 1, "probabilities": {"done": 1}}},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    monkeypatch.setenv("PERPLEXITY_DECIDER_MODEL", "pplx-decider-flash-latest")
    assert decider_model() == "pplx-decider-flash-latest"
    client = _client(handler)
    client.system_one(state={"goal": "go"}, questions={"kind": Choice("which", {"done": "stop"})})
    assert seen["model"] == "pplx-decider-flash-latest"
    client.system_one(
        state={"goal": "go"},
        questions={"kind": Choice("which", {"done": "stop"})},
        model="pplx-decider-v1.2",
    )
    assert seen["model"] == "pplx-decider-v1.2"


def test_an_error_keeps_the_request_id_and_drops_the_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            headers={"x-request-id": "req_bad"},
            json={"error": {"message": f"rejected {KEY}", "type": "invalid_request_error", "code": "400"}},
        )

    with pytest.raises(DecisionError) as caught:
        _client(handler).system_one(state={"goal": "go"}, questions={"kind": Choice("which", {"done": "stop"})})
    text = str(caught.value)
    assert caught.value.request_id == "req_bad"
    assert "req_bad" in text
    assert "invalid_request_error" in text
    assert "rejected" in text
    assert KEY not in text


def test_missing_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(DecisionError, match="PERPLEXITY_API_KEY"):
        PerplexityDecider()


def request_text(body: dict) -> str:
    return json.dumps(body)
