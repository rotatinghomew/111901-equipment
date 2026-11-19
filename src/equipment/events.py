"""故障事件聚合。

归属一个事件同时考虑四个维度：物理设备、发生时间、部件（类别）、
已核实的修复生效边界。

边界规则：
- status=verified 的处置，其 repair_effective 是硬边界：
  提交时间 >= 生效日的报障属于修后新事件，不得并入旧单；
- status=pending 的处置不构成边界，只作为弱证据挂在当前开放事件上；
- 同一开放事件内的催问（chase）与同类别描述记为重复反映；
- 类别冲突的描述只建立"疑似关联（provisional）"，不确认合并，
  不进入官方合并统计。

跨设备编号的合并只发生在 identity 模块确认的同一物理单元内。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .classify import Classification
from .identity import IdentityResolver
from .models import Asset, Disposition, Report

RELATION_ANCHOR = "anchor"
RELATION_FOLLOWUP = "duplicate_followup"        # 催问：同一事件的重复反映
RELATION_DUPLICATE = "duplicate_description"    # 同类别近似重复件
RELATION_CONFLICT = "provisional_conflict"      # 类别冲突：疑似关联，未确认
CONFIRMED_RELATIONS = {RELATION_ANCHOR, RELATION_FOLLOWUP, RELATION_DUPLICATE}


@dataclass(frozen=True)
class ReportLink:
    report: str
    asset: str
    submitted_at: date
    source: str
    category: str | None
    chase: bool
    uncertain: bool
    relation: str
    note: str


@dataclass(frozen=True)
class FaultEvent:
    event_id: str
    unit_id: str
    assets: tuple[str, ...]
    family: str
    window_index: int
    opened_at: date
    closed_at: date | None                     # None 表示仍在开放
    status: str                                # open / verified_closed
    links: tuple[ReportLink, ...]
    pending_dispositions: tuple[Disposition, ...]
    verified_category: str | None
    provisional: bool

    @property
    def official_links(self) -> tuple[ReportLink, ...]:
        """官方统计只承认已确认关系；疑似关联单列。"""
        return tuple(link for link in self.links if link.relation in CONFIRMED_RELATIONS)

    @property
    def provisional_links(self) -> tuple[ReportLink, ...]:
        return tuple(link for link in self.links if link.relation == RELATION_CONFLICT)

    @property
    def chase_count(self) -> int:
        return sum(1 for link in self.links if link.relation == RELATION_FOLLOWUP)

    @property
    def anchor(self) -> ReportLink:
        return next(link for link in self.links if link.relation == RELATION_ANCHOR)


class EventBuilder:
    def __init__(self, assets: list[Asset], reports: list[Report],
                 dispositions: list[Disposition], identity: IdentityResolver,
                 classifications: dict[str, Classification]):
        self.assets = {a.asset: a for a in assets}
        self.reports = sorted(reports, key=lambda r: (r.submitted_at, r.report))
        self.dispositions = dispositions
        self.identity = identity
        self.classifications = classifications

    def build(self) -> list[FaultEvent]:
        events: list[FaultEvent] = []
        for unit in self.identity.units:
            events.extend(self._build_for_unit(unit.unit_id, unit.assets, unit.family,
                                               unit.confirmed))
        return sorted(events, key=lambda e: (e.opened_at, e.event_id))

    def _build_for_unit(self, unit_id: str, members: tuple[str, ...],
                        family: str, unit_confirmed: bool) -> list[FaultEvent]:
        member_set = set(members)
        unit_reports = [r for r in self.reports if r.asset in member_set]
        unit_dispositions = [d for d in self.dispositions if d.asset in member_set]
        if not unit_reports:
            return []

        verified_bounds = sorted(
            {d.repair_effective for d in unit_dispositions
             if d.is_verified and d.repair_effective is not None}
        )
        verified_by_date: dict[date, Disposition] = {
            d.repair_effective: d for d in unit_dispositions if d.is_verified
        }
        pending = [d for d in unit_dispositions if not d.is_verified]

        def window_of(day: date) -> int:
            index = 0
            for bound in verified_bounds:
                if day >= bound:
                    index += 1
                else:
                    break
            return index

        buckets: dict[int, list[Report]] = {}
        for report in unit_reports:
            buckets.setdefault(window_of(report.submitted_at), []).append(report)

        events: list[FaultEvent] = []
        for window_index in sorted(buckets):
            window_reports = sorted(buckets[window_index],
                                    key=lambda r: (r.submitted_at, r.report))
            opened_at = window_reports[0].submitted_at
            # 窗口 k 的左边界即第 k 个核实修复日；窗口 0 无左边界。
            left_bound = verified_bounds[window_index - 1] if window_index > 0 else None
            # 窗口 k 若存在第 k+1 个核实修复日，则该处置关闭本窗口。
            has_closing = window_index < len(verified_bounds)
            right_bound = verified_bounds[window_index] if has_closing else None
            closing_disp = verified_by_date[right_bound] if has_closing else None
            status = "verified_closed" if has_closing else "open"

            links: list[ReportLink] = []
            anchor_category: str | None = None
            provisional = not unit_confirmed
            for report in window_reports:
                cls = self.classifications[report.report]
                if not links:
                    relation = RELATION_ANCHOR
                    anchor_category = cls.category
                    note = "窗口内首条描述，作为事件锚点"
                elif cls.chase:
                    relation = RELATION_FOLLOWUP
                    note = "催问件，按同一事件的重复反映挂载"
                elif cls.category is not None and cls.category == anchor_category:
                    relation = RELATION_DUPLICATE
                    note = "同部件同类别描述，按近似重复件挂载"
                else:
                    relation = RELATION_CONFLICT
                    provisional = True
                    note = "类别不一致或未判定，疑似关联但未确认合单"
                if not unit_confirmed:
                    note += "；物理单元迁移关系未确认，跨编号合单被冻结"
                links.append(ReportLink(
                    report=report.report, asset=report.asset,
                    submitted_at=report.submitted_at, source=report.source,
                    category=cls.category, chase=cls.chase,
                    uncertain=cls.uncertain, relation=relation, note=note,
                ))

            window_pending = tuple(
                d for d in pending
                if (left_bound is None
                    or d.repair_effective is None
                    or d.repair_effective >= left_bound)
                and (right_bound is None
                     or d.repair_effective is None
                     or d.repair_effective < right_bound)
            )
            events.append(FaultEvent(
                event_id=f"{unit_id}-w{window_index:02d}",
                unit_id=unit_id, assets=members, family=family,
                window_index=window_index, opened_at=opened_at,
                closed_at=right_bound,
                status=status, links=tuple(links),
                pending_dispositions=window_pending,
                verified_category=closing_disp.category if closing_disp else None,
                provisional=provisional,
            ))
        return events


def recurrence_by_family(events: list[FaultEvent]) -> dict[str, dict[str, float]]:
    """重复故障率：同一物理单元在核实修复后再次出现事件即计复发。

    分子=发生复发的单元数，分母=有事件的单元数。疑似关联不触发复发计数。
    """
    units_by_family: dict[str, set[str]] = {}
    recurrent: dict[str, set[str]] = {}
    for event in events:
        units_by_family.setdefault(event.family, set()).add(event.unit_id)
        if event.window_index > 0 and not event.provisional:
            recurrent.setdefault(event.family, set()).add(event.unit_id)
    result: dict[str, dict[str, float]] = {}
    for family, units in units_by_family.items():
        rec = len(recurrent.get(family, set()))
        result[family] = {
            "active_units": len(units),
            "recurrent_units": rec,
            "recurrence_rate": rec / len(units),
        }
    return dict(sorted(result.items()))
