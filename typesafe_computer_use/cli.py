"""Command-line entry points: `clicker` and `clicker-inspect`."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from . import config
from .actions import Context
from .perception import capture, perceive
from .platform_adapter import desktop
from .report import annotate, ax_count, render_payload
from .runner import RunConfig, run
from .supplied import SuppliedError, supplied_text, supplied_url
from .timing import format_timing

DOTENV = Path.cwd() / ".env"


def _prepare() -> None:
    config.load_dotenv(DOTENV)
    if not os.environ.get("PERPLEXITY_API_KEY"):
        sys.exit("PERPLEXITY_API_KEY is not set (export it or put it in .env)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="clicker",
        description="Drive this computer toward a goal: screen OCR, a Perplexity decision model, deterministic actions.",
    )
    parser.add_argument("goal", help="what you want done on this computer")
    parser.add_argument("--act", action="store_true", help="actually click and type (default: dry run, one step)")
    parser.add_argument("--text", help="exact text to type when the decision is type_text; typed unchanged")
    parser.add_argument("--text-file", type=Path, help="file whose exact UTF-8 contents are typed, including a trailing newline")
    parser.add_argument("--submit", action="store_true", help="press Return after the supplied text")
    parser.add_argument("--open-url", help="exact https URL to open when the decision is a site outside the catalog")
    parser.add_argument("--steps", type=int, default=config.DEFAULT_STEPS, help="max actions before stopping")
    parser.add_argument("--min-confidence", type=float, default=config.DEFAULT_MIN_CONFIDENCE, help="stop below this confidence")
    parser.add_argument("--delay", type=float, default=config.DEFAULT_DELAY, help="seconds to wait after each action")
    parser.add_argument(
        "--handoffs",
        type=int,
        default=config.DEFAULT_HANDOFFS,
        help="kept for compatibility; clicker never calls an answer model, so every stop is final",
    )
    parser.add_argument("--out", type=Path, default=Path("runs") / time.strftime("%Y%m%d-%H%M%S"), help="run folder")
    parser.add_argument("--image", type=Path, help="replay a saved capture instead of the live screen (never acts)")
    parser.add_argument("--app", help="frontmost app to report during replay")
    parser.add_argument("--url", help="browser URL to report during replay")
    args = parser.parse_args(argv)

    _prepare()
    if args.act and not desktop.accessibility_trusted():
        sys.exit("this terminal lacks Accessibility permission; grant it in System Settings > Privacy & Security")
    try:
        text = supplied_text(args.text, args.text_file)
        open_url = supplied_url(args.open_url)
    except SuppliedError as e:
        sys.exit(str(e))
    if args.submit and text is None:
        sys.exit("--submit needs --text, --text-file, CLICKER_TEXT, or CLICKER_TEXT_FILE")
    if text is None:
        print("no text supplied: type_text is not offered (pass --text or --text-file)")
    else:
        print(f"text supplied ({len(text)} characters)" + (", then Return" if args.submit else ""))
    if open_url is None:
        print("no URL supplied: a site outside the catalog cannot be opened")
    else:
        print("open URL supplied")

    cfg = RunConfig(
        goal=args.goal,
        out=args.out,
        act=args.act,
        steps=args.steps,
        min_confidence=args.min_confidence,
        delay=args.delay,
        handoffs=args.handoffs,
        image=args.image,
        app=args.app,
        url=args.url,
    )

    def ctx_factory(decider, history):
        return Context(
            goal=args.goal,
            browser=config.browser(),
            email=config.email(),
            decider=decider,
            writer=None,
            history=history,
            ask=None,
            text=text,
            submit=args.submit,
            open_url=open_url,
        )

    state = run(cfg, ctx_factory)
    if state.outcome.startswith("aborted"):
        sys.exit(130)


def inspect(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="clicker-inspect",
        description="Count down, capture the screen, and show the facts and criteria the clicker would send.",
    )
    parser.add_argument("goal", nargs="?", default="(no goal given)")
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--no-open", action="store_true", help="write files without opening them")
    parser.add_argument("--out", type=Path, default=Path("inspections") / time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args(argv)
    config.load_dotenv(DOTENV)
    args.out.mkdir(parents=True, exist_ok=True)

    for n in range(args.countdown, 0, -1):
        print(f"{n}...", end=" ", flush=True)
        time.sleep(1)
    print("capture")

    browser = config.browser()
    timing: dict[str, float] = {}
    screen = capture(browser=browser, timing=timing)
    items = perceive(screen, config.MAX_OPTIONS, args.goal, timing)
    annotated = args.out / "annotated.png"
    text = args.out / "state.txt"
    screen.image.save(args.out / "raw.png")
    annotate(screen, items, chosen="", out=annotated)
    text.write_text(render_payload(args.goal, screen, items, [], browser, config.email()), encoding="utf-8")

    print(
        f"app={screen.app!r} url={screen.url!r} items={len(items)} ax={ax_count(items)} "
        f"offscreen={len(screen.offscreen)} field={screen.field.role if screen.field else None}"
    )
    print(format_timing(timing))
    print(f"  {annotated}\n  {text}")
    if not args.no_open:
        desktop.open_path(annotated)
        desktop.open_path(text, as_text=True)
