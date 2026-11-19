"""故障类别口径的版本化管理。

关键约束：
- 每份描述按其发生日期适用当时生效的口径版本；
- 换版后老标签只做解释（lineage），不按新细类反向硬拆；
- 版本之间的映射只能落到父类或同义类，细类需要新证据支持。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class CategoryInfo:
    code: str
    name: str
    version: str
    safety_relevant: bool
    skill: str
    parent: str | None


class Taxonomy:
    def __init__(self, policy: dict[str, Any]):
        self.policy = policy
        self.current_version = policy["current_version"]
        self._versions = {v["version"]: v for v in policy["versions"]}
        self._ranges: list[tuple[date, date | None, str]] = []
        for version in policy["versions"]:
            start = date.fromisoformat(version["effective_from"])
            end_raw = version.get("effective_to")
            end = date.fromisoformat(end_raw) if end_raw else None
            self._ranges.append((start, end, version["version"]))
        self._ranges.sort()

    def version_for(self, day: date) -> str:
        """返回某日期适用的口径版本。"""
        for start, end, version in self._ranges:
            if day >= start and (end is None or day <= end):
                return version
        raise ValueError(f"日期 {day} 不在任何口径版本的生效区间内")

    def version_window(self, version: str) -> tuple[date, date | None]:
        spec = self._versions[version]
        start = date.fromisoformat(spec["effective_from"])
        end_raw = spec.get("effective_to")
        return start, date.fromisoformat(end_raw) if end_raw else None

    def category(self, code: str, version: str | None = None) -> CategoryInfo:
        """按指定版本（默认当前版）解释一个类别码。"""
        version = version or self.current_version
        spec = self._versions[version]
        for cat in spec["categories"]:
            if cat["code"] == code:
                return CategoryInfo(
                    code=cat["code"],
                    name=cat["name"],
                    version=version,
                    safety_relevant=bool(cat.get("safety_relevant", False)),
                    skill=cat.get("skill", "inspection"),
                    parent=cat.get("parent"),
                )
        raise KeyError(f"口径 {version} 中不存在类别 {code}")

    def is_safety(self, code: str | None, version: str | None = None) -> bool:
        if not code:
            return False
        return self.category(code, version).safety_relevant

    def skill_for(self, code: str | None) -> str:
        """技能需求按当前版口径解析；未判定项走通用 inspection。"""
        if not code:
            return self.policy.get("abstain_skill", "inspection")
        return self.category(code).skill

    def explain_label(self, code: str, recorded_version: str) -> dict[str, str]:
        """解释一个历史标签在当前口径下的含义，不做硬拆。"""
        historical = self.category(code, recorded_version)
        current_spec = self._versions[self.current_version]
        current_codes = {c["code"] for c in current_spec["categories"]}
        lineage = current_spec.get("lineage", {}).get(code)
        if code in current_codes:
            current = self.category(code)
            mapping = "retained"
        elif historical.parent and historical.parent in current_codes:
            current = self.category(historical.parent)
            mapping = "rolled_to_parent"
        else:
            current = self.category(code, recorded_version)
            mapping = "version_bound"
        return {
            "code": code,
            "recorded_version": recorded_version,
            "recorded_name": historical.name,
            "current_code": current.code,
            "current_name": current.name,
            "mapping": mapping,
            "lineage_note": lineage["interpretation"] if lineage else "无换版映射记录，按原版本口径保留。",
        }

    def rules_for(self, version: str) -> list[dict[str, Any]]:
        return list(self._versions[version]["rules"])

    def uncertainty_markers(self) -> list[str]:
        return list(self.policy["uncertainty_markers"])

    def impact_markers(self) -> list[dict[str, str]]:
        return list(self.policy["impact_markers"])

    def lineage_codes(self, version: str) -> set[str]:
        return set(self._versions[version].get("lineage", {}).keys())
