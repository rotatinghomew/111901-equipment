"""现场复核名额分配。

两类并行的抽核轨道，用途严格区分：

- 风险派发（risk）：按误分后果、安全疑似、未覆盖型号、催问量等加权，
  入样机会未知，结果只能定性使用，禁止外推总体错误率或直接排班；
- 轮换抽样（rotation）：分层随机、入样机会（pi）随抽样动作记录，
  只有这一轨的复核结果可用于外推整个器材群的错误率。

硬约束：
- 疑似安全故障为强制现场项，名额不足时进入升级清单，
  绝不因抽核名额用完而标记为"可继续开放"；
- 名额按 地点×技能 的周名额扣除已约人员后计算；
- 已约人员、地点与所需技能同时参与分配。
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from .events import FaultEvent, RELATION_CONFLICT
from .models import Route

STREAM_RISK = "risk"
STREAM_ROTATION = "rotation"
STREAM_ESCALATED = "escalated"
STREAM_DEFERRED = "deferred"


@dataclass(frozen=True)
class ReviewUnit:
    unit_key: str
    event_id: str
    asset: str
    site: str
    family: str
    skill_required: str
    safety_suspected: bool
    priority_score: float
    score_breakdown: dict[str, float]
    rationale: tuple[str, ...]
    unseen_family: bool
    chase_count: int
    has_conflict: bool
    has_abstain: bool


@dataclass(frozen=True)
class Assignment:
    unit_key: str
    event_id: str
    site: str
    skill_required: str
    stream: str
    route: str | None
    inclusion_probability: float | None
    note: str


@dataclass(frozen=True)
class AllocationResult:
    assignments: tuple[Assignment, ...]
    escalated: tuple[Assignment, ...]
    deferred: tuple[Assignment, ...]
    capacity_remaining: dict[tuple[str, str], int]
    rotation_design: dict[str, object]

    def by_stream(self, stream: str) -> tuple[Assignment, ...]:
        return tuple(a for a in self.assignments if a.stream == stream)


class CapacityLedger:
    """地点×技能 的周名额台账，扣除已约人员。"""

    def __init__(self, routes: list[Route], bookings: list[dict[str, object]] | None = None):
        self.capacity: dict[tuple[str, str], int] = {}
        self.route_by_key: dict[tuple[str, str], str] = {}
        for route in routes:
            key = (route.site, route.skill)
            self.capacity[key] = self.capacity.get(key, 0) + route.weekly_slots
            # 同地点同技能若有多条路线，合并名额并记录全部路线。
            self.route_by_key.setdefault(key, route.route)
        for booking in bookings or []:
            key = (str(booking["site"]), str(booking["skill"]))
            count = int(booking.get("slots", 1))
            if key not in self.capacity:
                raise ValueError(f"预约指向无路线的 地点/技能: {key}")
            self.capacity[key] -= count
            if self.capacity[key] < 0:
                raise ValueError(f"预约数超过周名额: {key}")

    def take(self, site: str, skill: str) -> str | None:
        key = (site, skill)
        if self.capacity.get(key, 0) <= 0:
            return None
        self.capacity[key] -= 1
        return self.route_by_key[key]

    def total_remaining(self) -> int:
        return sum(v for v in self.capacity.values() if v > 0)


class ReviewPlanner:
    def __init__(self, sampling_policy: dict, skill_lookup, *, seed: int | None = None):
        self.weights = sampling_policy["priority_weights"]
        self.min_rotation_fraction = sampling_policy["min_rotation_fraction"]
        self.seed = seed if seed is not None else sampling_policy["rotation_seed"]
        self.skill_lookup = skill_lookup

    def build_units(self, events: list[FaultEvent], *,
                    safety_event_ids: set[str],
                    unseen_families: set[str],
                    site_by_asset: dict[str, str]) -> list[ReviewUnit]:
        """每个待判事件只生成一份证据需求（一个复核单元）。"""
        units: list[ReviewUnit] = []
        for event in events:
            if event.status == "verified_closed" and not event.provisional:
                continue
            anchor = event.anchor
            skill = self.skill_lookup(anchor.category)
            chase = event.chase_count
            has_conflict = any(link.relation == RELATION_CONFLICT for link in event.links)
            has_abstain = any(link.category is None for link in event.links)
            safety = event.event_id in safety_event_ids
            unseen = event.family in unseen_families

            breakdown = {
                "safety": self.weights["safety"] * (1.0 if safety else 0.0),
                "mismerge_consequence": self.weights["mismerge_consequence"]
                                        * (1.0 if has_conflict else 0.0),
                "uncertain": self.weights["uncertain"]
                             * (1.0 if (has_abstain or any(
                                 link.uncertain for link in event.links)) else 0.0),
                "unseen_family": self.weights["unseen_family"] * (1.0 if unseen else 0.0),
                "chase_volume": self.weights["chase_volume"] * min(1.0, chase / 3.0),
            }
            rationale: list[str] = []
            if safety:
                rationale.append("疑似安全故障，强制现场复核")
            if has_conflict:
                rationale.append("含类别冲突的疑似合单，误合会影响巡查频率保持")
            if has_abstain:
                rationale.append("含未判定描述，漏合会抬高该型号重复故障率")
            if unseen:
                rationale.append(f"型号 {event.family} 未被训练覆盖")
            if chase:
                rationale.append(f"含 {chase} 条催问，高频催问可能挤占巡检")
            if not rationale:
                rationale.append("开放事件需一份现场证据关闭")

            units.append(ReviewUnit(
                unit_key=f"ru-{event.event_id}",
                event_id=event.event_id,
                asset=anchor.asset,
                site=site_by_asset[anchor.asset],
                family=event.family,
                skill_required=skill,
                safety_suspected=safety,
                priority_score=round(sum(breakdown.values()), 6),
                score_breakdown={k: round(v, 6) for k, v in breakdown.items()},
                rationale=tuple(rationale),
                unseen_family=unseen,
                chase_count=chase,
                has_conflict=has_conflict,
                has_abstain=has_abstain,
            ))
        return units

    def allocate(self, units: list[ReviewUnit], ledger: CapacityLedger) -> AllocationResult:
        rng = random.Random(self.seed)
        units_by_key = {u.unit_key: u for u in units}

        # 第一轨：疑似安全故障强制派发，名额不足即升级，不得开放。
        assignments: list[Assignment] = []
        escalated: list[Assignment] = []
        safety_units = [u for u in units if u.safety_suspected]
        other_units = [u for u in units if not u.safety_suspected]
        for unit in safety_units:
            route = ledger.take(unit.site, unit.skill_required)
            if route is None:
                escalated.append(Assignment(
                    unit.unit_key, unit.event_id, unit.site, unit.skill_required,
                    STREAM_ESCALATED, None, None,
                    "疑似安全故障无可用名额，升级处理，禁止标记为可继续开放",
                ))
            else:
                assignments.append(Assignment(
                    unit.unit_key, unit.event_id, unit.site, unit.skill_required,
                    STREAM_RISK, route, None, "安全强制项，按风险派发",
                ))

        # 第二轨：轮换抽样，按型号分层；记录每层入样机会 pi。
        eligible = list(other_units)
        strata: dict[str, list[ReviewUnit]] = defaultdict(list)
        for unit in eligible:
            strata[unit.family].append(unit)

        total_capacity = ledger.total_remaining()
        # 安全强制项已固定走风险轨，轮换目标只受非安全池与剩余名额约束。
        rotation_target = min(
            len(eligible),
            round(total_capacity * self.min_rotation_fraction) if total_capacity else 0,
        )
        rotation_selected: set[str] = set()
        rotation_design: dict[str, object] = {"strata": {}, "seed": self.seed}
        # 按层规模比例分配名额，合计不超过 rotation_target；目标 > 0 时
        # 每个有成员的层至少 1 个，最后一层吸收取整误差。
        strata_items = sorted(strata.items())
        layer_quota: dict[str, int] = {}
        remaining_target = rotation_target
        for index, (family, members) in enumerate(strata_items):
            if index == len(strata_items) - 1:
                quota = min(len(members), max(0, remaining_target))
            elif rotation_target == 0:
                quota = 0
            else:
                proportional = rotation_target * len(members) / max(1, len(eligible))
                quota = min(len(members), max(1, round(proportional)))
                quota = min(quota, max(0, remaining_target))
            layer_quota[family] = quota
            remaining_target -= quota
        for family, members in strata_items:
            quota = max(0, min(layer_quota.get(family, 0), len(members)))
            pool = sorted(members, key=lambda u: u.unit_key)
            picks = rng.sample(pool, quota) if quota else []
            pi = quota / len(members) if members else 0.0
            rotation_design["strata"][family] = {
                "eligible": len(members), "sampled": quota,
                "inclusion_probability": round(pi, 6),
            }
            for pick in picks:
                rotation_selected.add(pick.unit_key)

        rotation_units = [u for u in eligible if u.unit_key in rotation_selected]
        risk_units = [u for u in eligible if u.unit_key not in rotation_selected]

        deferred: list[Assignment] = []
        for unit in rotation_units:
            route = ledger.take(unit.site, unit.skill_required)
            if route is None:
                deferred.append(Assignment(
                    unit.unit_key, unit.event_id, unit.site, unit.skill_required,
                    STREAM_DEFERRED, None, None,
                    "轮换抽中但 地点/技能 名额不足，本期顺延并保留入样机会",
                ))
                continue
            family_design = rotation_design["strata"][unit.family]
            assignments.append(Assignment(
                unit.unit_key, unit.event_id, unit.site, unit.skill_required,
                STREAM_ROTATION, route,
                family_design["inclusion_probability"],
                f"轮换抽样（型号 {unit.family} 分层），入样机会已记录",
            ))

        # 第三轨：剩余名额按风险优先级派发，pi 未知。
        for unit in sorted(risk_units, key=lambda u: (-u.priority_score, u.unit_key)):
            route = ledger.take(unit.site, unit.skill_required)
            if route is None:
                deferred.append(Assignment(
                    unit.unit_key, unit.event_id, unit.site, unit.skill_required,
                    STREAM_DEFERRED, None, None,
                    "风险项但本期名额不足，顺延下一期，不作为可开放结论",
                ))
                continue
            assignments.append(Assignment(
                unit.unit_key, unit.event_id, unit.site, unit.skill_required,
                STREAM_RISK, route, None,
                "风险派发：" + "；".join(unit.rationale),
            ))

        return AllocationResult(
            assignments=tuple(assignments),
            escalated=tuple(escalated),
            deferred=tuple(deferred),
            capacity_remaining=dict(ledger.capacity),
            rotation_design=rotation_design,
        )


def extrapolate_error_rate(assignments: list[Assignment],
                           reviewed_outcomes: dict[str, bool]) -> dict[str, object]:
    """仅用轮换样本做 Horvitz-Thompson 外推。

    reviewed_outcomes: unit_key -> True 表示复核发现错误。
    任何轮换单元缺少入样机会时返回 not_estimable；风险轨样本一律剔除。
    """
    rotation = [a for a in assignments if a.stream == STREAM_ROTATION]
    missing = [a.unit_key for a in rotation if a.unit_key not in reviewed_outcomes]
    if missing or not rotation:
        return {
            "estimable": False,
            "reason": "轮换样本缺少复核结果或轮换样本为空，外推不成立",
            "risk_stream_excluded": True,
        }
    if any(a.inclusion_probability in (None, 0.0) for a in rotation):
        return {
            "estimable": False,
            "reason": "存在入样机会未知的轮换单元，禁止外推",
            "risk_stream_excluded": True,
        }
    ht_total = sum(1.0 / (a.inclusion_probability or 1.0) for a in rotation)
    ht_errors = sum(
        (1.0 / (a.inclusion_probability or 1.0))
        for a in rotation if reviewed_outcomes[a.unit_key]
    )
    return {
        "estimable": True,
        "estimated_population_size": round(ht_total, 4),
        "estimated_errors": round(ht_errors, 4),
        "estimated_error_rate": round(ht_errors / ht_total, 4),
        "sample_size": len(rotation),
        "risk_stream_excluded": True,
        "note": "仅轮换轨参与外推；风险轨入样机会未知，已剔除。",
    }
