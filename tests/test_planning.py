import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from equipment.classify import RuleBasedClassifier
from equipment.events import EventBuilder
from equipment.identity import IdentityResolver
from equipment.io import DataBundle
from equipment.models import Asset, Disposition, Report
from equipment.planning import (LedgerEntry, PlanningLedger, ReauditService,
                                ReviewOutcome, supervisor_view)
from equipment.review import (CapacityLedger, ReviewPlanner, STREAM_RISK,
                              STREAM_ROTATION)
from equipment.taxonomy import Taxonomy


class LedgerImmutabilityTest(unittest.TestCase):
    def test_entries_append_only_and_supersede_keeps_hours(self):
        ledger = PlanningLedger()
        old = LedgerEntry("t1", date(2026, 7, 1), "s1", "mechanical", 2.0,
                          "e1", "按当时待巡清单派发")
        ledger.record_task(old)
        with self.assertRaises(ValueError):
            ledger.record_task(old)
        ledger.record_task(LedgerEntry("t2", date(2026, 7, 8), "s1", "mechanical", 1.5,
                                       "e2", "复核确认修后新事件，重新派发"))
        ledger.mark_superseded("t1", "t2")
        # 历史条目仍在，时数与依据保留
        self.assertEqual(len(ledger.entries), 2)
        self.assertEqual(ledger.entries[0].hours, 2.0)
        self.assertEqual(ledger.entries[0].basis, "按当时待巡清单派发")
        # 有效在账时数只统计未被替换的任务
        self.assertEqual(ledger.booked_hours("s1", "mechanical"), 1.5)


class ReauditFlowTest(unittest.TestCase):
    def setUp(self):
        self.assets = [Asset("a1", "f1", "s1", date(2024, 1, 1), None, None)]
        self.reports = [
            Report("r1", "a1", date(2026, 7, 1), "异响需要复查", "web"),
            Report("r2", "a1", date(2026, 7, 10), "异响需要复查", "web"),
        ]
        self.dispositions = [
            Disposition("a1", "e1", date(2026, 7, 5), "c0", "verified"),
        ]
        taxonomy = Taxonomy(DataBundle(ROOT / "data").taxonomy)
        identity = IdentityResolver(self.assets)
        classifications = {
            c.report: c for c in RuleBasedClassifier(taxonomy).classify_many(
                self.reports, {"a1": "f1"})
        }
        self.events = EventBuilder(self.assets, self.reports, self.dispositions,
                                   identity, classifications).build()
        self.taxonomy = taxonomy

    def _allocation(self, stream, rotation_fraction=0.3):
        policy = dict(DataBundle(ROOT / "data").sampling)
        policy["min_rotation_fraction"] = rotation_fraction
        planner = ReviewPlanner(policy, self.taxonomy.skill_for)
        units = planner.build_units(
            self.events, safety_event_ids=set(), unseen_families=set(),
            site_by_asset={"a1": "s1"})
        from equipment.models import Route
        ledger = CapacityLedger([Route("s1", "rt1", "mechanical", 5)])
        result = planner.allocate(units, ledger)
        return result

    def test_new_fault_after_repair_refreshes_watchlist_and_replaces_weak_evidence(self):
        ledger = PlanningLedger()
        ledger.record_task(LedgerEntry("t1", date(2026, 7, 2), "s1", "mechanical", 2.0,
                                       self.events[0].event_id, "旧事件巡查"))
        allocation = self._allocation(STREAM_ROTATION)
        service = ReauditService(ledger)
        new_event = self.events[1]
        outcome = ReviewOutcome(
            unit_key=f"ru-{new_event.event_id}", event_id=new_event.event_id,
            reviewed_at=date(2026, 7, 12), finding="new_fault_after_repair",
            notes="现场确认修复后再次异响")
        reaudit = service.apply_outcome(outcome, new_event,
                                        all_events=self.events, allocation=allocation)
        self.assertTrue(any("新事件" in s for s in reaudit.watchlist_additions))
        self.assertIn("f1", reaudit.family_scope)
        self.assertTrue(reaudit.recurrence_refresh["f1"]["recurrent_units"] >= 1)
        # 历史时数保留
        self.assertEqual(ledger.entries[0].hours, 2.0)

    def test_risk_stream_conclusion_is_marked_not_schedulable(self):
        allocation = self._allocation(STREAM_RISK, rotation_fraction=0.0)
        # 构造一个必然落在风险轨的结果
        target = next(a for a in allocation.assignments if a.stream == STREAM_RISK)
        event = next(e for e in self.events if e.event_id == target.event_id)
        service = ReauditService(PlanningLedger())
        reaudit = service.apply_outcome(
            ReviewOutcome(target.unit_key, target.event_id, date(2026, 7, 12),
                          "confirmed_fault", "核实"),
            event, all_events=self.events, allocation=allocation)
        self.assertTrue(any("不可直接排班" in s for s in reaudit.schedule_restrictions))

    def test_supervisor_view_lists_reasons_coverage_and_restrictions(self):
        ledger = PlanningLedger()
        allocation = self._allocation(STREAM_ROTATION)
        view = supervisor_view(
            events=self.events, allocation=allocation, reaudits=[], ledger=ledger,
            coverage_months=("2026-07",), covered_families={"f1"})
        self.assertIn("2026-07", view["coverage"]["months"])
        self.assertIn("f1", view["coverage"]["families"])
        self.assertIn("patrol_increase_reasons", view)
        self.assertIn("not_schedulable_conclusions", view)
        self.assertIn("recurrence_by_family", view)


if __name__ == "__main__":
    unittest.main()
