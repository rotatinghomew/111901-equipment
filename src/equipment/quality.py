"""分类与合并质量的离线评估。

两条独立验收口径，避免学习与验收互相污染：

1. 时间后移切片：以事件为单位按开放时间切分，同一事件及其近似重复件
   整体落入同一侧，训练侧与验收侧单号、事件号不得相交；
2. 型号留出：训练分类器未覆盖过的 family 单独成片，度量未判定
   （abstain）范围，不与错误率混算。

另提供 Wilson 区间与泄漏断言；重点排查样本的入样机会未知，
其误差不得外推（外推逻辑在 review 模块的轮换抽样中实现）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from .classify import Classification, RuleBasedClassifier
from .events import FaultEvent
from .models import Report
from .taxonomy import Taxonomy


@dataclass(frozen=True)
class RateEstimate:
    numerator: int
    denominator: int
    point: float
    ci_low: float
    ci_high: float

    def as_dict(self) -> dict[str, object]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "point": round(self.point, 4),
            "ci95": [round(self.ci_low, 4), round(self.ci_high, 4)],
        }


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> RateEstimate:
    if trials == 0:
        return RateEstimate(0, 0, 0.0, 0.0, 0.0)
    phat = successes / trials
    denom = 1 + z * z / trials
    center = (phat + z * z / (2 * trials)) / denom
    half = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denom
    return RateEstimate(successes, trials, phat, max(0.0, center - half), min(1.0, center + half))


@dataclass(frozen=True)
class EventSplit:
    train_events: tuple[FaultEvent, ...]
    test_events: tuple[FaultEvent, ...]
    train_reports: frozenset[str]
    test_reports: frozenset[str]

    def assert_no_leakage(self) -> None:
        overlap = self.train_reports & self.test_reports
        if overlap:
            raise AssertionError(f"报障单号同时出现在训练与验收侧: {sorted(overlap)}")
        train_event_ids = {e.event_id for e in self.train_events}
        test_event_ids = {e.event_id for e in self.test_events}
        shared = train_event_ids & test_event_ids
        if shared:
            raise AssertionError(f"事件同时出现在训练与验收侧: {sorted(shared)}")


def temporal_event_split(events: list[FaultEvent], cutoff: date) -> EventSplit:
    """按事件开放日期切分；同一事件的锚点与重复件天然整体同侧。"""
    train = tuple(e for e in events if e.opened_at < cutoff)
    test = tuple(e for e in events if e.opened_at >= cutoff)
    return EventSplit(
        train_events=train,
        test_events=test,
        train_reports=frozenset(link.report for e in train for link in e.links),
        test_reports=frozenset(link.report for e in test for link in e.links),
    )


def family_holdout_split(events: list[FaultEvent], held_out_families: set[str]
                         ) -> tuple[tuple[FaultEvent, ...], tuple[FaultEvent, ...]]:
    covered = tuple(e for e in events if e.family not in held_out_families)
    uncovered = tuple(e for e in events if e.family in held_out_families)
    return covered, uncovered


def build_ground_truth(events: list[FaultEvent]) -> dict[str, str]:
    """真值只取自核实关闭事件的处置类别。

    开放事件、pending 处置、疑似关联链接均不产生真值标签。
    """
    truth: dict[str, str] = {}
    for event in events:
        if event.status != "verified_closed" or not event.verified_category:
            continue
        for link in event.official_links:
            truth[link.report] = event.verified_category
    return truth


@dataclass(frozen=True)
class ClassificationScore:
    scored: int
    correct: int
    wrong: int
    abstained: int
    unadjudicated: int
    error_rate: RateEstimate
    abstain_rate: RateEstimate
    mismatches: tuple[tuple[str, str | None, str], ...]


def score_classifier(classifier: RuleBasedClassifier,
                     reports: list[Report],
                     ground_truth: dict[str, str],
                     family_by_asset: dict[str, str]) -> ClassificationScore:
    results: list[Classification] = classifier.classify_many(reports, family_by_asset)
    correct = wrong = abstained = unadjudicated = 0
    mismatches: list[tuple[str, str | None, str]] = []
    for result in results:
        truth = ground_truth.get(result.report)
        if truth is None:
            unadjudicated += 1
            continue
        if result.category is None:
            abstained += 1
            continue
        if result.category == truth:
            correct += 1
        else:
            wrong += 1
            mismatches.append((result.report, result.category, truth))
    scored = correct + wrong
    return ClassificationScore(
        scored=scored,
        correct=correct,
        wrong=wrong,
        abstained=abstained,
        unadjudicated=unadjudicated,
        error_rate=wilson_interval(wrong, scored),
        abstain_rate=wilson_interval(abstained, scored + abstained),
        mismatches=tuple(mismatches),
    )


@dataclass(frozen=True)
class MergeScore:
    true_pairs: int
    predicted_pairs: int
    correct_pairs: int
    merge_precision: RateEstimate
    merge_recall: RateEstimate
    provisional_links: int
    provisional_in_official_stats: int


def _pair_count(group_sizes: list[int]) -> int:
    return sum(n * (n - 1) // 2 for n in group_sizes)


def score_merges(events: list[FaultEvent]) -> MergeScore:
    """以核实窗口为真值分组，评估已确认合单的成对精确率/召回率。

    精确率/召回率只在有真值的核实关闭事件上计算；开放事件没有真值，
    其已确认合单不计入分母，疑似关联仅作全局监控计数。
    """
    true_groups: dict[str, list[str]] = {}
    predicted_groups: dict[str, list[str]] = {}
    provisional_links = 0
    provisional_leak = 0

    for event in events:
        provisional_links += len(event.provisional_links)
        provisional_leak += sum(
            1 for link in event.provisional_links if link in event.official_links
        )
        if event.status != "verified_closed":
            continue
        # 真值：同一物理单元、同一核实窗口内的全部报障。
        true_groups[event.event_id] = sorted(link.report for link in event.links)
        if not event.provisional:
            predicted_groups[event.event_id] = sorted(
                link.report for link in event.official_links
            )

    def pair_set(groups: dict[str, list[str]]) -> set[frozenset[str]]:
        pairs: set[frozenset[str]] = set()
        for members in groups.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pairs.add(frozenset((members[i], members[j])))
        return pairs

    truth_pairs = pair_set(true_groups)
    predicted_pairs = pair_set(predicted_groups)
    correct = len(truth_pairs & predicted_pairs)
    return MergeScore(
        true_pairs=len(truth_pairs),
        predicted_pairs=len(predicted_pairs),
        correct_pairs=correct,
        merge_precision=wilson_interval(correct, len(predicted_pairs)),
        merge_recall=wilson_interval(correct, len(truth_pairs)),
        provisional_links=provisional_links,
        provisional_in_official_stats=provisional_leak,
    )


def train_scoped_classifier(taxonomy: Taxonomy,
                            train_events: list[FaultEvent]) -> RuleBasedClassifier:
    """在训练侧事件覆盖的型号上"训练"（规则体系下即限定适用范围）。"""
    families = {event.family for event in train_events}
    return RuleBasedClassifier(taxonomy, scoped_families=families)
