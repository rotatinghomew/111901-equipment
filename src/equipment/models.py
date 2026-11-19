"""领域值对象。

所有记录均为只读；空字符串按"无登记值"处理，不推断业务事实。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Asset:
    asset: str
    family: str
    site: str
    commissioned: date | None
    moved_from: str | None
    moved_at: date | None


@dataclass(frozen=True)
class Report:
    report: str
    asset: str
    submitted_at: date
    text: str
    source: str


@dataclass(frozen=True)
class Disposition:
    asset: str
    event: str
    repair_effective: date | None
    category: str
    status: str

    @property
    def is_verified(self) -> bool:
        return self.status == "verified"


@dataclass(frozen=True)
class Route:
    site: str
    route: str
    skill: str
    weekly_slots: int
