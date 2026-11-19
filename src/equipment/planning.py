"""复核结论回写与巡视安排重审。

设计原则：
- 已核销巡查时数按当时任务原样保留，台账只追加、不修改（immutable append-only）；
- 复核确认修后新事件时，待巡清单、重复故障率、同型号观察范围一起重审；
- 被替换的弱证据（pending 处置、疑似关联、未判定描述）必须显式说明；
- 区域主管可查询加巡理由、覆盖的型号/月份，以及哪些结论仍不可用于排班。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .events import FaultEvent, recurrence_by_family
from .review import AllocationResult, STREAM_RISK, STREAM_ROTATION


@dataclass(frozen=True)
class LedgerEntry:
    task_id: str
    recorded_at: date
    site: str
    skill: str
    hours: float
    event_id: str
    basis: str                       # 当时依据，历史任务不可改写
    superseded_by: str | None = None


@dataclass(frozen=True)
class ReviewOutcome:
    unit_key: str
    event_id: str
    reviewed_at: date
    finding: str                     # confirmed_fault / new_fault_after_repair / not_reproduced / mismerged
    notes: str


@dataclass(frozen=True)
class ReauditResult:
    outcome: ReviewOutcome
    watchlist_additions: tuple[str, ...]
    watchlist_removals: tuple[str, ...]
    replaced_weak_evidence: tuple[str, ...]
    recurrence_refresh: dict[str, dict[str, float]]
    family_scope: tuple[str, ...]
    schedule_restrictions: tuple[str, ...]


class PlanningLedger:
    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []
        self._superseded: set[str] = set()

    def record_task(self, entry: LedgerEntry) -> None:
        if any(e.task_id == entry.task_id for e in self._entries):
            raise ValueError(f"任务编号已存在，台账不可覆盖: {entry.task_id}")
        self._entries.append(entry)

    def mark_superseded(self, task_id: str, superseding_task_id: str) -> None:
        """历史条目不删除不改写，仅登记被新任务替换的指针。"""
        if not any(e.task_id == task_id for e in self._entries):
            raise KeyError(f"未知任务编号: {task_id}")
        self._superseded.add(task_id)

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    def booked_hours(self, site: str, skill: str) -> float:
        return sum(e.hours for e in self._entries
                   if e.site == site and e.skill == skill
                   and e.task_id not in self._superseded)


class ReauditService:
    def __init__(self, ledger: PlanningLedger):
        self.ledger = ledger

    def apply_outcome(self, outcome: ReviewOutcome, event: FaultEvent,
                      *, all_events: list[FaultEvent],
                      allocation: AllocationResult) -> ReauditResult:
        additions: list[str] = []
        removals: list[str] = []
        replaced: list[str] = []
        restrictions: list[str] = []

        if outcome.finding == "new_fault_after_repair":
            additions.append(
                f"{event.event_id} 经现场复核确认是修复生效后的新事件，"
                f"加入待巡清单，不得与修复前旧单合并"
            )
            for disposition in event.pending_dispositions:
                replaced.append(
                    f"pending 处置 {disposition.asset}/{disposition.event} 的弱边界假设"
                    f"被现场复核替换（复核日期 {outcome.reviewed_at}）"
                )
            for link in event.links:
                if link.category is None:
                    replaced.append(
                        f"未判定描述 {link.report} 的弱证据由现场结论替换"
                    )
        elif outcome.finding == "mismerged":
            removals.append(f"{event.event_id} 的疑似合单被拆回，从重复故障率分子中移除")
            replaced.append("事件内 provisional 关联经复核判定为错误合单，已拆分")
        elif outcome.finding == "not_reproduced":
            removals.append(f"{event.event_id} 现场未复现，移出待巡清单但保留报告痕迹")
        else:
            replaced.append(f"{event.event_id} 故障经现场确认，开放推断转为核实结论")

        refreshed = recurrence_by_family(all_events)
        family_scope = tuple(sorted({
            e.family for e in all_events
            if any(link.report in {l.report for l in event.links} for link in e.links)
        })) or (event.family,)

        risk_keys = {a.unit_key for a in allocation.by_stream(STREAM_RISK)}
        if outcome.unit_key in risk_keys:
            restrictions.append(
                "该复核来自风险派发轨，入样机会未知，结论不可外推总体错误率、不可直接排班"
            )
        if event.provisional:
            restrictions.append("事件仍含未确认迁移/合单关系，统计仅列疑似口径")
        if any(a.inclusion_probability is None for a in allocation.assignments):
            restrictions.append("存在无入样机会的抽核件，相关错误率结论暂不能用于排班")

        return ReauditResult(
            outcome=outcome,
            watchlist_additions=tuple(additions),
            watchlist_removals=tuple(removals),
            replaced_weak_evidence=tuple(replaced),
            recurrence_refresh=refreshed,
            family_scope=family_scope,
            schedule_restrictions=tuple(restrictions),
        )


def supervisor_view(*, events: list[FaultEvent],
                    allocation: AllocationResult,
                    reaudits: list[ReauditResult],
                    ledger: PlanningLedger,
                    coverage_months: tuple[str, ...],
                    covered_families: set[str]) -> dict[str, object]:
    """区域主管一页式可追溯视图。"""
    reasons: list[str] = []
    for assignment in allocation.assignments:
        if assignment.stream == STREAM_RISK:
            reasons.append(f"{assignment.event_id}: {assignment.note}")
    for reaudit in reaudits:
        reasons.extend(reaudit.watchlist_additions)

    not_schedulable: list[str] = []
    for reaudit in reaudits:
        not_schedulable.extend(reaudit.schedule_restrictions)
    if not any(a for a in allocation.by_stream(STREAM_ROTATION)):
        not_schedulable.append("本期无轮换抽样单元，总体错误率无法外推，排班仍按现行频率")

    return {
        "patrol_increase_reasons": reasons,
        "coverage": {
            "months": list(coverage_months),
            "families": sorted(covered_families),
            "rotation_strata": allocation.rotation_design.get("strata", {}),
        },
        "recurrence_by_family": recurrence_by_family(events),
        "not_schedulable_conclusions": sorted(set(not_schedulable)),
        "booked_hours_kept": [
            {
                "task_id": e.task_id, "site": e.site, "skill": e.skill,
                "hours": e.hours, "basis": e.basis,
                "superseded": e.task_id in ledger._superseded,
            }
            for e in ledger.entries
        ],
        "escalated_safety": [
            {"event_id": a.event_id, "site": a.site, "note": a.note}
            for a in allocation.escalated
        ],
    }
