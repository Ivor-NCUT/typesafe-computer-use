<p align="center">
  <img src="docs/banner.svg" alt="typesafe-computer-use" width="100%">
</p>

<p align="center">
  <a href="https://github.com/awlevin/typesafe-computer-use/actions/workflows/ci.yaml"><img alt="CI" src="https://github.com/awlevin/typesafe-computer-use/actions/workflows/ci.yaml/badge.svg"></a>
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white">
  <img alt="macOS, Windows experimental" src="https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20(experimental)-000000">
  <a href="https://docs.perplexity.ai/docs/decisions/quickstart"><img alt="Perplexity Decisions" src="https://img.shields.io/badge/decisions-Perplexity%20Decisions%20API-20808d"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
</p>

**typesafe-computer-use** (jev for short) drives a Mac toward a goal you type in plain English.
It reads the screen deterministically, asks Perplexity's Decisions API which action comes
next, and only calls a writing model when a text field genuinely needs free text or the
decision model has stopped and the screen needs reading.

One idea runs through it: the classifier picks, code decides facts, and the writer only writes
free text. Anything a model would have to work out (a date, whether a field is focused, whether
a URL is clean) is computed in code and handed over as state. [VISION.md](VISION.md) says what
it is for in a dozen lines.

```
clicker "go to techcrunch and take me to the checkout page for the cheapest tickets to their next upcoming event" --act
```

> **Beta.** This is under heavy development. Expect rough edges, and expect settings and
> behavior to change between 0.x [releases](https://github.com/awlevin/typesafe-computer-use/releases).
> It drives your real mouse and keyboard, so start with a dry run.

## Why a classifier

Frontier-model computer use is capable and expensive: every step ships a screenshot and
waits several seconds for a plan. Most steps do not need a plan. They need one choice
from a short list, made quickly and cheaply, with a confidence number you can gate on.

[Perplexity's Decisions API](https://docs.perplexity.ai/docs/decisions/quickstart) is that
model. It answers a `choice` over up to 255 options with a probability for each option
and a confidence, or a `noul` yes/no with the probability of yes. It does not write
text, generate code, or explain itself. This loop still decides the facts and executes
the action. The default model is `pplx-decider-latest` ($0.02 per million input tokens,
output tokens free). Set `PERPLEXITY_DECIDER_MODEL` to `pplx-decider-flash-latest` for
the smaller model. The API reads images, so each decision sends the screenshot as a PNG
data URL next to the OCR text, accessibility labels, and the other facts code computed.

A frontier model can read a date off the pixels and compare it unaided. This loop still
computes dates, rows, focus, and the actions already tried on this screen in code, and
hands them over as state, as described in [how a step works](docs/how-a-step-works.md).
The screenshot is there so the model can also see what the text missed.

This project began as Aaron Levin's
[typesafe-computer-use](https://github.com/awlevin/typesafe-computer-use) (MIT). The
computer-use loop is that project. The decision model here is Perplexity's.

## Install

macOS 14 or newer, Python 3.12 or newer, [uv](https://docs.astral.sh/uv/). Windows 10 and 11
are experimental; see [Windows](docs/windows.md).

```
git clone https://github.com/awlevin/typesafe-computer-use
cd typesafe-computer-use
uv sync
cp .env.example .env     # fill in the keys
```

| variable | required | purpose |
|---|---|---|
| `PERPLEXITY_API_KEY` | yes | every decision (`Authorization: Bearer`) |
| `PERPLEXITY_DECIDER_MODEL` | no | defaults to `pplx-decider-latest` |
| `ANTHROPIC_API_KEY` | no | `type_text`, writer-proposed URLs, and the final answer |
| `CLICKER_EMAIL` | no | enables the `type_email` action |
| `CLICKER_BROWSER` | no | defaults to `Google Chrome` |

`.env` lives at the repo root and is read by every entry point (`clicker`, `clicker-inspect`).
To run the writer on another model or endpoint (LM Studio, Ollama, any OpenAI-compatible
server), see [writer endpoints](docs/writer-endpoints.md).

Grant your terminal **Screen Recording** and **Accessibility** in System Settings >
Privacy & Security. Without the first, captures are wallpaper. Without the second,
synthetic clicks are silently dropped, and `--act` refuses to start.

## First run

```
uv run clicker "open the Playground"                 # dry run: one step, prints what it would do
uv run clicker "open the Playground" --act           # drives the machine, up to 100 steps
uv run clicker "log in" --act --steps 20 --delay 3   # longer and slower
uv run clicker "log in" --act --handoffs 0           # the classifier alone: its first stop ends the run
uv run clicker-inspect "any goal"                    # 3-2-1, capture, open the annotated screen + payload
```

Clear the terminal first. It is on screen, so its text is OCR input.

To stop a live run, press Ctrl-C in the terminal or slam the mouse into the top-left corner of
the screen. The loop also stops itself on `done`, low confidence, a stall, or the step limit,
and the writer then reads the screen and prints the answer. Every run writes
`runs/<timestamp>/`, so a stall can be [replayed offline](docs/run-folder.md).

## How a step works

1. The screen is read deterministically: Vision OCR on a crop of the frontmost window, plus the
   labelled controls from the accessibility tree, which sees the icons OCR cannot.
2. Code adds the facts: the date on any block and how far off it is, the row of a repeated
   label, the focused field, the app and URL, and the actions already tried on this screen.
3. One Decisions API request answers three `choice` questions — which kind of action, which item, which site — and includes the screenshot.
4. The action runs deterministically, the loop waits, and the next capture is the only witness
   of what it did.
5. When the classifier stops, the writer reads the screen and answers, or hands the run back
   with a focus (one move) or a question for you. It never picks an action.

The full walk, with the OCR cost, the tree walk, the action space, and the stop rules, is in
[how a step works](docs/how-a-step-works.md).

## Benchmark

jev runs as an agent in [OSWorld](https://github.com/xlang-ai/OSWorld-V2), a benchmark of real
desktop tasks, beside OSWorld's own GPT agent on the same task. `scripts/osworld setup` fetches
OSWorld and installs jev beside it; `scripts/osworld run-jev chrome/<task id> --ocr rapidocr`
runs one task. It needs a Linux host with KVM; see [OSWorld](docs/osworld.md) for that and for
a one-command machine in Google Cloud.

## Read more

- [How a step works](docs/how-a-step-works.md): perception, the three-part decision, the action space, stalls, and the hand-off to the writer
- [Run folder](docs/run-folder.md): what every run writes, the timing line, and offline replay
- [Browser backend](docs/browser-backend.md): DOM perception over Chrome DevTools, no OCR, no screen permission
- [Writer endpoints](docs/writer-endpoints.md): every writer variable, and other models over the Anthropic or OpenAI API
- [Windows](docs/windows.md): the experimental adapter and how it differs from macOS
- [OSWorld](docs/osworld.md): setup, the two run commands, results, and the Google Cloud machine
- [Layout](docs/layout.md): every module and what it owns
- [Known limits](docs/known-limits.md)
- [Sandbox](docs/sandbox.md): a Linux computer in a container, for running the agent on a screen that is not yours

## Contributing

Bug reports with a run folder attached are the most useful thing you can send. Before a pull
request, `uv run ruff check . && uv run ruff format --check .` and `uv run pytest -q` must pass.
The ground rules, and how to write a scenario for a task the loop cannot do, are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE). Copyright 2026 Aaron Levin. This fork keeps that license and that attribution.
