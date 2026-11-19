"""报障描述的规则化分类。

输出不仅是类别，还包含：
- 判断凭据（命中规则、命中词、适用口径版本）；
- 不确定项（疑似/待复查等标记）；
- 影响服务程度；
- 是否为催问（chase）而非新故障描述；
- 未判定（abstain）时类别为空，交由抽核决定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .models import Report
from .taxonomy import Taxonomy


@dataclass(frozen=True)
class Classification:
    report: str
    text: str
    source: str
    submitted_at: date
    taxonomy_version: str
    category: str | None          # None 表示未判定（abstain）
    chase: bool
    uncertain: bool
    impact: str                   # none / service_degraded / service_affected
    safety_suspected: bool
    evidence: list[str] = field(default_factory=tuple)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # frozen dataclass 下把 list 转 tuple 保持不可变
        object.__setattr__(self, "evidence", tuple(self.evidence))


class RuleBasedClassifier:
    """按描述发生时适用的口径版本执行关键词规则。"""

    def __init__(self, taxonomy: Taxonomy, *, scoped_families: set[str] | None = None):
        self.taxonomy = taxonomy
        # 留出型号（held-out family）上的分类器必须弃权，用于度量未判定范围。
        self.scoped_families = scoped_families

    def classify(self, report: Report, family: str | None = None) -> Classification:
        if self.scoped_families is not None and family not in self.scoped_families:
            return self._abstain(report, reason="held_out_family")

        version = self.taxonomy.version_for(report.submitted_at)
        text = report.text
        evidence: list[str] = [f"适用口径版本={version}"]

        uncertain_markers = [
            marker for marker in self.taxonomy.uncertainty_markers() if marker in text
        ]
        uncertain = bool(uncertain_markers)
        if uncertain_markers:
            evidence.append(f"不确定标记={','.join(uncertain_markers)}")

        matched = self._match_rules(text, version)
        if matched is None:
            return self._abstain(report, reason="no_rule_matched", evidence=evidence,
                                 uncertain=uncertain, version=version)

        rule_id, category, is_chase = matched
        evidence.append(f"命中规则={rule_id}")
        impact = self._impact(text, is_chase)
        evidence.append(f"影响程度依据={impact}")

        safety = bool(category and self.taxonomy.is_safety(category, version) and uncertain)
        if safety:
            evidence.append("安全相关类别且描述含不确定标记→疑似安全故障")

        return Classification(
            report=report.report,
            text=text,
            source=report.source,
            submitted_at=report.submitted_at,
            taxonomy_version=version,
            category=category,
            chase=is_chase,
            uncertain=uncertain,
            impact=impact,
            safety_suspected=safety,
            evidence=evidence,
        )

    def classify_many(self, reports: list[Report],
                      family_by_asset: dict[str, str] | None = None) -> list[Classification]:
        family_by_asset = family_by_asset or {}
        return [self.classify(r, family_by_asset.get(r.asset)) for r in reports]

    def _match_rules(self, text: str, version: str) -> tuple[str, str | None, bool] | None:
        for rule in self.taxonomy.rules_for(version):
            if "all" in rule and not all(token in text for token in rule["all"]):
                continue
            if "any" in rule and not any(token in text for token in rule["any"]):
                continue
            return rule["id"], rule.get("category"), bool(rule.get("chase", False))
        return None

    def _impact(self, text: str, is_chase: bool) -> str:
        if is_chase:
            return self.taxonomy.policy["chase_impact"]
        levels = self.taxonomy.policy["impact_levels_ordered"]
        level = self.taxonomy.policy["default_impact"]
        for marker in self.taxonomy.impact_markers():
            if marker["marker"] in text:
                candidate = marker["level"]
                if levels.index(candidate) > levels.index(level):
                    level = candidate
        return level

    def _abstain(self, report: Report, *, reason: str, evidence: list[str] | None = None,
                 uncertain: bool = False, version: str | None = None) -> Classification:
        evidence = list(evidence or [])
        evidence.append(f"未判定原因={reason}")
        return Classification(
            report=report.report,
            text=report.text,
            source=report.source,
            submitted_at=report.submitted_at,
            taxonomy_version=version or self.taxonomy.version_for(report.submitted_at),
            category=None,
            chase=False,
            uncertain=uncertain,
            impact="none",
            safety_suspected=False,
            evidence=evidence,
        )
