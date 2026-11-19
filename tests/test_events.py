import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from equipment.classify import RuleBasedClassifier
from equipment.events import (EventBuilder, RELATION_ANCHOR, RELATION_CONFLICT,
                              RELATION_DUPLICATE, RELATION_FOLLOWUP,
                              recurrence_by_family)
from equipment.identity import IdentityResolver
from equipment.io import DataBundle
from equipment.models import Asset, Disposition, Report
from equipment.taxonomy import Taxonomy


def build_events(assets, reports, dispositions):
    taxonomy = Taxonomy(DataBundle(ROOT / "data").taxonomy)
    identity = IdentityResolver(assets)
    classifications = {
        c.report: c
        for c in RuleBasedClassifier(taxonomy).classify_many(
            reports, {a.asset: a.family for a in assets})
    }
    return EventBuilder(assets, reports, dispositions, identity, classifications).build()


def asset(aid, family="f1", site="s1", moved_from=None, moved_at=None):
    return Asset(aid, family, site, date(2024, 1, 1), moved_from, moved_at)


def report(rid, aid, day, text, source="web"):
    return Report(rid, aid, date.fromisoformat(day), text, source)


def disposition(aid, event, effective, category, status="verified"):
    return Disposition(aid, event, date.fromisoformat(effective) if effective else None,
                       category, status)


class EventBoundaryTest(unittest.TestCase):
    def test_verified_repair_splits_new_event_after_effective_date(self):
        events = build_events(
            [asset("a1")],
            [report("r1", "a1", "2026-07-01", "异响需要复查"),
             report("r2", "a1", "2026-07-10", "异响需要复查")],
            [disposition("a1", "e1", "2026-07-05", "c0")],
        )
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].status, "verified_closed")
        self.assertEqual(events[0].closed_at, date(2026, 7, 5))
        self.assertEqual(events[1].status, "open")
        self.assertEqual(events[1].window_index, 1)
        self.assertEqual([l.report for l in events[1].links], ["r2"])

    def test_report_on_effective_date_belongs_to_new_event(self):
        events = build_events(
            [asset("a1")],
            [report("r1", "a1", "2026-07-05", "异响需要复查")],
            [disposition("a1", "e1", "2026-07-05", "c0")],
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].window_index, 1)

    def test_pending_disposition_is_not_a_boundary(self):
        events = build_events(
            [asset("a1")],
            [report("r1", "a1", "2026-07-01", "异响需要复查"),
             report("r2", "a1", "2026-07-08", "持续催问是否修好")],
            [disposition("a1", "e1", "2026-07-05", "c0", status="pending")],
        )
        self.assertEqual(len(events), 1, "pending 不能划定修复生效边界")
        self.assertEqual(events[0].status, "open")
        relations = {l.report: l.relation for l in events[0].links}
        self.assertEqual(relations["r1"], RELATION_ANCHOR)
        self.assertEqual(relations["r2"], RELATION_FOLLOWUP)
        self.assertEqual(len(events[0].pending_dispositions), 1)

    def test_category_conflict_is_provisional_not_merged_in_official_stats(self):
        events = build_events(
            [asset("a1")],
            [report("r1", "a1", "2026-07-01", "异响需要复查"),
             report("r2", "a1", "2026-07-02", "座椅松动影响使用")],
            [],
        )
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertTrue(event.provisional)
        self.assertEqual(event.provisional_links[0].report, "r2")
        self.assertEqual(event.provisional_links[0].relation, RELATION_CONFLICT)
        self.assertNotIn("r2", [l.report for l in event.official_links])

    def test_same_category_reports_are_duplicate_descriptions(self):
        events = build_events(
            [asset("a1")],
            [report("r1", "a1", "2026-07-01", "异响需要复查"),
             report("r2", "a1", "2026-07-02", "异响需要复查", source="operator")],
            [],
        )
        relations = {l.report: l.relation for l in events[0].links}
        self.assertEqual(relations["r2"], RELATION_DUPLICATE)

    def test_cross_asset_merge_only_within_confirmed_physical_unit(self):
        events = build_events(
            [asset("n1", moved_from="old9", moved_at=date(2026, 6, 1)),
             asset("n2", moved_from="n1", moved_at=date(2026, 6, 2))],
            [report("r1", "n1", "2026-07-01", "异响需要复查"),
             report("r2", "n2", "2026-07-02", "异响需要复查")],
            [],
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(set(events[0].assets), {"n1", "n2"})

    def test_recurrence_rate_counts_only_confirmed_repeats(self):
        events = build_events(
            [asset("a1"), asset("a2", family="f2")],
            [report("r1", "a1", "2026-07-01", "异响需要复查"),
             report("r2", "a1", "2026-07-10", "异响需要复查"),
             report("r3", "a2", "2026-07-01", "座椅松动影响使用")],
            [disposition("a1", "e1", "2026-07-05", "c0")],
        )
        rates = recurrence_by_family(events)
        self.assertEqual(rates["f1"]["recurrent_units"], 1)
        self.assertEqual(rates["f1"]["recurrence_rate"], 1.0)
        self.assertEqual(rates["f2"]["recurrence_rate"], 0.0)

    def test_fixture_provisional_never_enters_official_stats(self):
        bundle = DataBundle(ROOT / "data")
        events = build_events(bundle.assets, bundle.reports, bundle.dispositions)
        for event in events:
            self.assertFalse(
                set(event.provisional_links) & set(event.official_links)
            )


if __name__ == "__main__":
    unittest.main()
