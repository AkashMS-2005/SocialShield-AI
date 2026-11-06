"""
adapters/base.py – Abstract base adapter for social media account normalisation.

Adheres to Working Rule #5:
  - No scraping or unofficial API bypass.
  - Normalises manual input or CSV rows into the common internal schema (AccountInput).
  - Provides a clearly documented placeholder interface where official authenticated
    APIs (e.g. Graph API, X API v2, YouTube Data API v3) can be plugged in later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from app.schemas import AccountInput, Platform


class BasePlatformAdapter(ABC):
    """
    Base class for platform adapters.
    Converts diverse platform-specific terminology into the unified AccountInput schema.
    """

    @property
    @abstractmethod
    def platform(self) -> Platform:
        """Returns the specific platform enum."""
        pass

    @abstractmethod
    def normalize_input(self, raw_data: Dict[str, Any]) -> AccountInput:
        """
        Normalises raw dictionary / CSV row with platform-specific column aliases
        into a validated AccountInput.
        """
        pass

    def fetch_from_official_api(self, username_or_id: str, credentials: Optional[Dict[str, str]] = None) -> AccountInput:
        """
        [FUTURE INTEGRATION PLACEHOLDER]
        When official API tokens / OAuth credentials are provided by the user,
        this method will connect to the official platform API (e.g. Meta Graph API,
        X API v2, YouTube Data API) via sanctioned endpoints without web scraping.
        """
        raise NotImplementedError(
            f"Official API client for {self.platform.value} is not connected. "
            "Input must be provided manually or via CSV upload (Rule #5)."
        )

