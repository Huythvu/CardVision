"""Render classification results into clean labels like A♠, 10♥, K♦."""

from __future__ import annotations

from . import config
from .classify import CardResult


def format_rank_suit(rank: str | None, suit: str | None, ascii_only: bool = False) -> str:
    """Format a rank+suit pair as e.g. 'A♠'. Missing values -> '??'."""
    if rank is None or suit is None:
        return "??"
    sym = suit if ascii_only else config.SUIT_SYMBOLS.get(suit, suit)
    return f"{rank}{sym}"


def format_card(result: CardResult, ascii_only: bool = False) -> str:
    """Format a CardResult as e.g. 'A♠'. Unknown/low-confidence -> '??'."""
    if not result.confident:
        return "??"
    return format_rank_suit(result.rank, result.suit, ascii_only)


def format_hand(results: dict[str, CardResult], ascii_only: bool = False) -> dict[str, str]:
    """Format a mapping of slot name -> CardResult into slot -> label."""
    return {slot: format_card(r, ascii_only) for slot, r in results.items()}
