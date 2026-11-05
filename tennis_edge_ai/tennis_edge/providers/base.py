"""Provider interface.

Every data source (historical files, CSV uploads, licensed APIs) implements
:class:`DataProvider`. A provider that is not configured reports
``available=False`` with a human-readable reason - the rest of the application
must then label the corresponding data as UNAVAILABLE rather than fabricating it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderStatus:
    name: str
    kind: str
    available: bool
    reason: str
    requires: list[str] = field(default_factory=list)
    terms_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"provider": self.name, "kind": self.kind, "available": self.available,
                "status": "AVAILABLE" if self.available else "UNAVAILABLE", "reason": self.reason,
                "requires": ", ".join(self.requires), "terms": self.terms_note}


class DataProvider(ABC):
    name: str = "base"
    kind: str = "generic"          # historical | projections | odds | results
    requires: list[str] = []
    terms_note: str = ""

    @abstractmethod
    def status(self) -> ProviderStatus:
        ...

    def _status(self, available: bool, reason: str) -> ProviderStatus:
        return ProviderStatus(self.name, self.kind, available, reason, list(self.requires), self.terms_note)


class ProviderUnavailable(RuntimeError):
    """Raised when a provider is asked for data it cannot legitimately supply."""
