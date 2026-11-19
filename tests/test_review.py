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
from equipment.models import Asset, Disposition, Report, Route
from equipment.pipeline import run_default
from equipment.review import (CapacityLedger, ReviewPlanner, STREAM_DEFERRED,
                              STREAM_ESCALATED, STREAM_RISK, STREAM_ROTATION,
                              extrapolate_error_rate)
from equipment.taxonomy import Taxonomy


def build_review_inputs(assets, reports, dispositions):
    taxonomy = Taxonomy(DataBundle(ROOT / "data").taxonomy)
    identity = IdentityResolver(assets)
    classifications = {
        c.report: c
        for c in RuleBasedClassifier(taxonomy).classify_many(
            reports, {a.asset: a.family for a in assets})
    }
    events = EventBuilder(assets, reports, dispositions, identity, classifications).build()
    safety = {
        event.event_id for event in events
        if any(classifications[link.report].safety_suspected for link in event.links)
    }
    return taxonomy, events, safety


class CapacityTest(unittest.TestCase):
    def test_bookings_reduce_capacity_and_overflow_rejected(self):
        routes = [Route("s1", "rt1", "mechanical", 3)]
        ledger = CapacityLedger(routes, bookings=[{"site": "s1", "skill": "mechanical",
                                                   "slots": 2}])
        self.assertEqual(ledger.take("s1", "mechanical"), "rt1")
        self.assertIsNone(ledger.take("s1", "mechanical"))

    def test_overbooked_ledger_raises(self):
        routes = [Route("s1", "rt1", "mechanical", 1)]
        with self.assertRaises(ValueError):
            CapacityLedger(routes, bookings=[{"site": "s1", "skill": "mechanical",
                                              "slots": 2}])

    def test_skill_mismatch_cannot_consume_capacity(self):
        ledger = CapacityLedger([Route("s1", "rt1", "mechanical", 2)])
        self.assertIsNone(ledger.take("s1", "electrical"))


class AllocationPolicyTest(unittest.TestCase):
    def setUp(self):
        self.assets = [
            Asset("a1", "f1", "s1", date(2024, 1, 1), None, None),
            Asset("a2", "f2", "s2", date(2024, 1, 1), None, None),
        ]
        self.reports = [
            Report("r1", "a1", date(2026, 7, 21), "疑似保护罩故障", "web"),
            Report("r2", "a2", date(2026, 7, 21), "异响需要复查", "web"),
        ]
        self.taxonomy, self.events, self.safety = build_review_inputs(
            self.assets, self.reports, [])
        self.policy = DataBundle(ROOT / "data").sampling
        self.planner = ReviewPlanner(self.policy, self.taxonomy.skill_for)

    def _units(self, unseen=None):
        return self.planner.build_units(
            self.events, safety_event_ids=self.safety,
            unseen_families=unseen or set(),
            site_by_asset={"a1": "s1", "a2": "s2"})

    def test_safety_suspected_is_forced_and_never_left_open(self):
        units = self._units()
        safety_unit = next(u for u in units if u.safety_suspected)
        self.assertEqual(safety_unit.skill_required, "inspection")
        # s1 inspection 零名额 → 必须升级，而不是被丢弃或标记可开放
        routes = [Route("s1", "rt1", "mechanical", 5),
                  Route("s2", "rt2", "inspection", 5)]
        result = self.planner.allocate(units, CapacityLedger(routes))
        self.assertEqual(len(result.escalated), 1)
        self.assertEqual(result.escalated[0].event_id, safety_unit.event_id)
        self.assertIn("禁止标记为可继续开放", result.escalated[0].note)
        self.assertNotIn(safety_unit.unit_key,
                         [d.unit_key for d in result.deferred])

    def test_safety_takes_capacity_before_other_streams(self):
        units = self._units()
        routes = [Route("s1", "rt1", "inspection", 1),
                  Route("s2", "rt2", "inspection", 5)]
        result = self.planner.allocate(units, CapacityLedger(routes))
        safety_assignment = next(a for a in result.assignments if a.stream == STREAM_RISK
                                 and a.event_id == units[0].event_id)
        self.assertEqual(safety_assignment.site, "s1")

    def test_rotation_records_inclusion_probability_and_risk_does_not(self):
        units = self._units(unseen={"f2"})
        policy = dict(self.policy)
        policy["min_rotation_fraction"] = 0.5
        planner = ReviewPlanner(policy, self.taxonomy.skill_for)
        routes = [Route("s1", "rt1", "inspection", 5),
                  Route("s2", "rt2", "mechanical", 5),
                  Route("s2", "rt3", "inspection", 5)]
        result = planner.allocate(units, CapacityLedger(routes))
        rotation = result.by_stream(STREAM_ROTATION)
        self.assertTrue(rotation)
        for assignment in rotation:
            self.assertIsNotNone(assignment.inclusion_probability)
            self.assertGreater(assignment.inclusion_probability, 0.0)
        for assignment in result.by_stream(STREAM_RISK):
            self.assertIsNone(assignment.inclusion_probability)

    def test_extrapolation_uses_only_rotation_stream(self):
        units = self._units(unseen={"f2"})
        policy = dict(self.policy)
        policy["min_rotation_fraction"] = 0.5
        planner = ReviewPlanner(policy, self.taxonomy.skill_for)
        routes = [Route("s1", "rt1", "inspection", 5),
                  Route("s2", "rt2", "mechanical", 5),
                  Route("s2", "rt3", "inspection", 5)]
        result = planner.allocate(units, CapacityLedger(routes))
        outcomes = {a.unit_key: True for a in result.assignments
                    if a.stream == STREAM_ROTATION}
        estimate = extrapolate_error_rate(list(result.assignments), outcomes)
        self.assertTrue(estimate["estimable"])
        self.assertTrue(estimate["risk_stream_excluded"])
        # 缺少复核结果 → 不可外推
        self.assertFalse(extrapolate_error_rate(list(result.assignments), {})["estimable"])

    def test_priority_score_reflects_mismerge_and_unseen_family(self):
        units = self._units(unseen={"f2"})
        by_family = {u.family: u for u in units}
        self.assertGreater(by_family["f2"].score_breakdown["unseen_family"], 0.0)
        self.assertGreater(by_family["f1"].score_breakdown["safety"], 0.0)

    def test_fixture_allocation_respects_total_capacity(self):
        result = run_default(ROOT / "data")
        allocation = result["review_allocation"]
        used = len(allocation["assignments"])
        total_capacity = sum(Route(**{
            "site": r.site, "route": r.route, "skill": r.skill,
            "weekly_slots": r.weekly_slots,
        }).weekly_slots for r in DataBundle(ROOT / "data").routes)
        self.assertLessEqual(used, total_capacity)
        # 双轨都存在时，轮换设计必须带分层入样机会
        if allocation["assignments"]:
            for layer in allocation["rotation_design"]["strata"].values():
                self.assertIn("inclusion_probability", layer)


if __name__ == "__main__":
    unittest.main()
