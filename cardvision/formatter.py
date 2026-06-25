"""Render classification results into clean labels like A♠, 10♥, K♦."""

from __future__ import annotations

from . import config
from .classify import CardResult


def format_card(result: CardResult, ascii_only: bool = False) -> str:
    """Format a CardResult as e.g. 'A♠'. Unknown/low-confidence -> '??'."""
    if not result.confident:
        return "??"
    suit = result.suit if ascii_only else config.SUIT_SYMBOLS.get(result.suit, result.suit)
    return f"{result.rank}{suit}"


def format_hand(results: dict[str, CardResult], ascii_only: bool = False) -> dict[str, str]:
    """Format a mapping of slot name -> CardResult into slot -> label."""
    return {slot: format_card(r, ascii_only) for slot, r in results.items()}
