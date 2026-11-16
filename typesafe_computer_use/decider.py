"""The decision client: Perplexity's Decisions API, one request per classifier call.

The loop still builds the questions and still executes the action. This module only
asks. A `choice` is one of up to 255 options, with a probability for each. A `noul`
is a yes/no. The model does not write text.

`state` is the facts the loop already computed, plus the screenshot when there is
one. The screenshot is an OpenAI-style `image_url` data URL. The API does not fetch
http(s) image URLs, and an image past 2048 tiles of 32x32 can hang and then 504, so
the image is resized here. At most 8 images go with a question. Web grounding stays
off: a screen action is decided from the screen, not from a search.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any, Protocol

from perplexity import APITimeoutError, Perplexity, PerplexityError
from PIL import Image

from .config import decider_model

TIMEOUT_SECONDS = 60.0  # an image past the tile limit hangs for about a minute, then the API returns 504
TILE = 32
MAX_TILES = 2048
MAX_IMAGES = 8


class DecisionError(RuntimeError):
    """A decision request that did not return answers. `request_id` is the API's x-request-id."""

    def __init__(self, message: str, *, request_id: str | None = None):
        self.request_id = request_id
        super().__init__(message if not request_id else f"{message} (x-request-id: {request_id})")


class Classifier(Protocol):
    """The one call the loop makes. Tests stand a fake in for it."""

    def system_one(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class Choice:
    """One option from `criteria`. The description may be empty; the name still counts."""

    instructions: str
    criteria: dict[str, str | None]


@dataclass(frozen=True)
class Noul:
    """Yes/no. At least one of `instructions` and `criteria` is required by the API."""

    instructions: str | None = None
    criteria: dict[str, str] | None = None


@dataclass(frozen=True)
class Score:
    """An ordered rubric. The loop does not ask this today; the client accepts it."""

    instructions: str | None
    criteria: list[str]


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class NoulAnswer:
    noul: float


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    confidence: float
    legend: dict[str, Any]
    probabilities: dict[str, float]


@dataclass(frozen=True)
class Reply:
    """What `system_one` hands back. `usage` has `input_tokens` and `output_tokens`."""

    answers: dict[str, ChoiceAnswer | NoulAnswer | ScoreAnswer]
    model: str
    usage: Any = None
    request_id: str | None = None


def tile_count(width: int, height: int) -> int:
    """Tiles after rounding each side to the nearest multiple of 32, which is how the API counts."""
    return round(width / TILE) * round(height / TILE)


def fits_tiles(width: int, height: int) -> bool:
    return width >= 1 and height >= 1 and tile_count(width, height) <= MAX_TILES


def fit_size(width: int, height: int) -> tuple[int, int]:
    """The largest size at or under the original that stays within 2048 tiles, aspect kept.

    1440x1440 and 2048x1024 already fit. 1600x1310 does not, and sending it can hang.
    """
    width, height = max(1, int(width)), max(1, int(height))
    if fits_tiles(width, height):
        return width, height
    scale = (MAX_TILES / ((width / TILE) * (height / TILE))) ** 0.5
    nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
    while not fits_tiles(nw, nh) and (nw > 1 or nh > 1):
        if nw >= nh and nw > 1:
            nw -= 1
        elif nh > 1:
            nh -= 1
        else:
            break
    return nw, nh


def image_part(image: Image.Image) -> dict[str, Any]:
    """One screenshot as a PNG data URL. PNG because screen text does not survive JPEG well."""
    fitted = image.convert("RGB")
    size = fit_size(*fitted.size)
    if fitted.size != size:
        fitted = fitted.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    fitted.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def _is_image_part(value: object) -> bool:
    if not isinstance(value, dict) or value.get("type") != "image_url":
        return False
    image_url = value.get("image_url")
    return isinstance(image_url, dict) and isinstance(image_url.get("url"), str)


def count_images(value: object) -> int:
    if _is_image_part(value):
        return 1
    if isinstance(value, dict):
        return sum(count_images(item) for item in value.values())
    if isinstance(value, list):
        return sum(count_images(item) for item in value)
    return 0


def with_screenshot(state: Any, image: Image.Image | None) -> Any:
    """Facts plus the screenshot, screenshot first so the model sees pixels before the list.

    The text dump of a run leaves the image out; the raw capture is the file beside it.
    A state that already holds 8 images is sent as it is: a ninth makes the request fail.
    """
    if image is None or count_images(state) >= MAX_IMAGES:
        return state
    part = image_part(image)
    if isinstance(state, dict):
        return {"screenshot": part, **state}
    if isinstance(state, list):
        return [part, *state]
    if isinstance(state, str):
        return [part, state]
    return {"screenshot": part, "facts": state}


def question_payload(question: Any) -> dict[str, Any]:
    """A loop question as the API's `type` / `instructions` / `criteria` object."""
    if isinstance(question, Choice):
        if not question.criteria:
            raise DecisionError("a choice question needs at least one option")
        return {"type": "choice", "instructions": question.instructions, "criteria": dict(question.criteria)}
    if isinstance(question, Noul):
        body: dict[str, Any] = {"type": "noul"}
        if question.instructions:
            body["instructions"] = question.instructions
        if question.criteria:
            body["criteria"] = dict(question.criteria)
        if len(body) == 1:
            raise DecisionError("a noul question needs instructions or criteria")
        return body
    if isinstance(question, Score):
        if not question.criteria:
            raise DecisionError("a score question needs at least one level")
        body = {"type": "score", "criteria": list(question.criteria)}
        if question.instructions:
            body["instructions"] = question.instructions
        return body
    if isinstance(question, dict) and question.get("type") in {"choice", "noul", "score"}:
        return dict(question)
    raise DecisionError(f"unsupported question: {type(question).__name__}")


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    dumped = getattr(value, "model_dump", None)
    if dumped is not None:
        data = dumped()
        if isinstance(data, dict):
            return data
    return dict(value)


def parse_answer(raw: Any) -> ChoiceAnswer | NoulAnswer | ScoreAnswer:
    data = raw if isinstance(raw, dict) or hasattr(raw, "type") else _mapping(raw)
    kind = data["type"] if isinstance(data, dict) else getattr(data, "type", None)
    read = data.get if isinstance(data, dict) else lambda name, default=None: getattr(data, name, default)
    if kind == "choice":
        probabilities = read("probabilities") or {}
        return ChoiceAnswer(
            choice=str(read("choice")),
            probabilities={str(key): float(value) for key, value in dict(probabilities).items()},
            confidence=float(read("confidence")),
        )
    if kind == "noul":
        return NoulAnswer(noul=float(read("noul")))
    if kind == "score":
        legend = read("legend") or {}
        probabilities = read("probabilities") or {}
        return ScoreAnswer(
            score=float(read("score")),
            confidence=float(read("confidence")),
            legend=dict(legend),
            probabilities={str(key): float(value) for key, value in dict(probabilities).items()},
        )
    raise DecisionError(f"unexpected decision answer type: {kind!r}")


class PerplexityDecider:
    """Posts to `/v1/decisions`. The key is `Authorization: Bearer`, from `PERPLEXITY_API_KEY`.

    `model` on a call overrides the client's model, which overrides `PERPLEXITY_DECIDER_MODEL`,
    which overrides `pplx-decider-latest`. `mode` is never set.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = TIMEOUT_SECONDS,
        client: Any | None = None,
    ) -> None:
        self.model = (model or "").strip() or decider_model()
        self.timeout = timeout
        if client is not None:
            self._sdk = client
            self._owns_client = False
            return
        key = api_key if api_key is not None else os.environ.get("PERPLEXITY_API_KEY")
        if not key:
            raise DecisionError("PERPLEXITY_API_KEY is not set")
        # The SDK sends Authorization: Bearer. x-api-key is not accepted by this API.
        self._sdk = Perplexity(api_key=key, timeout=timeout)
        self._owns_client = True

    def close(self) -> None:
        if self._owns_client:
            self._sdk.close()

    def __enter__(self) -> PerplexityDecider:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False

    def system_one(self, *, state: Any, questions: dict, model: str | None = None, image: Image.Image | None = None) -> Reply:
        """One Decisions API request. `image` is folded into `state` and is not logged."""
        chosen = (model or "").strip() or self.model
        wire_questions = {name: question_payload(question) for name, question in questions.items()}
        if not wire_questions:
            raise DecisionError("a decision needs at least one question")
        wire_state = with_screenshot(state, image)
        if count_images(wire_state) > MAX_IMAGES:
            raise DecisionError(f"a decision can include at most {MAX_IMAGES} images")
        try:
            parsed, request_id = self._post(chosen, wire_state, wire_questions)
        except DecisionError:
            raise
        except Exception as exc:
            raise self._error(exc) from None
        answers = getattr(parsed, "answers", None)
        if not isinstance(answers, dict) or not answers:
            raise DecisionError("Perplexity decisions returned no answers", request_id=request_id)
        return Reply(
            answers={name: parse_answer(answer) for name, answer in answers.items()},
            model=str(getattr(parsed, "model", None) or chosen),
            usage=getattr(parsed, "usage", None),
            request_id=request_id,
        )

    def _post(self, model: str, state: Any, questions: dict[str, Any]) -> tuple[Any, str | None]:
        raw = self._sdk.decisions.with_raw_response.create(
            model=model,
            state=state,
            questions=questions,
            timeout=self.timeout,
        )
        request_id = raw.http_response.headers.get("x-request-id")
        return raw.parse(), request_id or None

    def _error(self, exc: Exception) -> DecisionError:
        """The API's message and type, plus x-request-id. The key is never part of the text."""
        request_id = None
        response = getattr(exc, "response", None)
        if response is not None:
            request_id = response.headers.get("x-request-id")
        message = getattr(exc, "message", None) or str(exc)
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict) and err.get("message"):
                message = str(err["message"])
                if err.get("type"):
                    message = f"{message}; type={err['type']}"
        status = getattr(exc, "status_code", None)
        prefix = f"{status}: " if isinstance(status, int) else ""
        if isinstance(exc, APITimeoutError):
            prefix = "timed out: "
        elif isinstance(exc, PerplexityError):
            prefix = prefix or "request failed: "
        return DecisionError(self._redact(f"Perplexity decisions failed ({prefix}{message})"), request_id=request_id)

    def _redact(self, text: str) -> str:
        key = getattr(self._sdk, "api_key", None)
        if isinstance(key, str) and key:
            text = text.replace(key, "[redacted]")
        return text
