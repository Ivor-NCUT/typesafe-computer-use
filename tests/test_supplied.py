"""The operator's exact text, and a disabled writer that never calls out."""

from types import SimpleNamespace

import pytest

from typesafe_computer_use.cli import main
from typesafe_computer_use.report import Log
from typesafe_computer_use.runner import RunConfig, RunState, hand_off
from typesafe_computer_use.supplied import SuppliedError, supplied_text, supplied_url
from typesafe_computer_use.writer import make_writer


def test_text_resolution_keeps_the_string_exactly(tmp_path, monkeypatch):
    path = tmp_path / "note.txt"
    path.write_bytes(b"  hello\n")
    assert supplied_text("  hello\n", None) == "  hello\n"
    assert supplied_text(None, path) == "  hello\n"
    monkeypatch.setenv("CLICKER_TEXT", "from env")
    assert supplied_text(None, None) == "from env"
    monkeypatch.delenv("CLICKER_TEXT")
    monkeypatch.setenv("CLICKER_TEXT_FILE", str(path))
    assert supplied_text(None, None) == "  hello\n"
    with pytest.raises(SuppliedError):
        supplied_text("a", path)


def test_a_url_flag_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("CLICKER_OPEN_URL", "https://from.env/")
    assert supplied_url("https://flag.example/") == "https://flag.example/"
    assert supplied_url(None) == "https://from.env/"


def test_make_writer_does_not_construct_a_client_without_a_key(clean_env, monkeypatch):
    monkeypatch.setattr("typesafe_computer_use.writer.anthropic.Anthropic", lambda *a, **k: pytest.fail("Anthropic client"))
    assert make_writer() is None


def test_clicker_uses_supplied_text_and_does_not_call_a_writer(clean_env, monkeypatch, tmp_path):
    """A key in the environment does not turn the writer on, and a stop does not call one."""
    monkeypatch.setenv("PERPLEXITY_API_KEY", "not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.setattr("typesafe_computer_use.writer.anthropic.Anthropic", lambda *a, **k: pytest.fail("Anthropic client"))
    seen = {}

    def fake_run(cfg, factory):
        seen["ctx"] = factory(None, [])
        seen["act"] = cfg.act
        return SimpleNamespace(outcome="done")

    monkeypatch.setattr("typesafe_computer_use.cli.run", fake_run)
    main(["search", "--text", "  hello\n", "--submit", "--open-url", "https://example.com/a", "--out", str(tmp_path)])

    ctx = seen["ctx"]
    assert seen["act"] is False
    assert ctx.writer is None and ctx.ask is None
    assert ctx.text == "  hello\n" and ctx.submit is True
    assert ctx.open_url == "https://example.com/a"
    assert hand_off(RunConfig(goal="search", out=tmp_path / "stop"), ctx, RunState(outcome="done"), 1, Log()) is False


def test_submit_without_text_is_refused(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "not-a-real-key")
    monkeypatch.setattr("typesafe_computer_use.cli.run", lambda *a, **k: pytest.fail("ran"))
    with pytest.raises(SystemExit):
        main(["search", "--submit", "--out", str(tmp_path)])
