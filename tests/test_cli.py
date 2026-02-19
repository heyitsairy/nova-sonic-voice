"""Tests for the CLI conversation display and argument parsing."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from nova_sonic.cli import ConversationDisplay, parse_args


class TestParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["cli"]):
            args = parse_args()
        assert args.voice == "matthew"
        assert args.system is None
        assert args.duration == 0
        assert args.debug is False

    def test_custom_voice(self):
        with patch("sys.argv", ["cli", "--voice", "ruth"]):
            args = parse_args()
        assert args.voice == "ruth"

    def test_custom_system_prompt(self):
        with patch("sys.argv", ["cli", "--system", "Be a pirate"]):
            args = parse_args()
        assert args.system == "Be a pirate"

    def test_duration(self):
        with patch("sys.argv", ["cli", "--duration", "60"]):
            args = parse_args()
        assert args.duration == 60

    def test_debug_flag(self):
        with patch("sys.argv", ["cli", "--debug"]):
            args = parse_args()
        assert args.debug is True


class TestConversationDisplay:
    def test_initial_state(self):
        display = ConversationDisplay()
        assert display._current_role is None
        assert display._turn_count == 0

    def test_on_user_text_sets_role(self, capsys):
        display = ConversationDisplay()
        display._start_time = time.time()
        display.on_user_text("user", "hello")
        assert display._current_role == "user"

    def test_on_assistant_text_increments_turns(self, capsys):
        display = ConversationDisplay()
        display._start_time = time.time()
        display.on_assistant_text("assistant", "hi there")
        assert display._turn_count == 1
        assert display._current_role == "assistant"

    def test_on_reconnect_resets_role(self, capsys):
        display = ConversationDisplay()
        display._start_time = time.time()
        display._current_role = "assistant"
        display.on_reconnect()
        assert display._current_role is None

    def test_start_prints_header(self, capsys):
        display = ConversationDisplay()
        display.start()
        captured = capsys.readouterr()
        assert "Nova 2 Sonic" in captured.out
        assert "Ctrl+C" in captured.out
