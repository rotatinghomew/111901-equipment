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
from equipment.quality import (build_ground_truth, family_holdout_split,
                              score_classifier, score_merges,
                              temporal_event_split, train_scoped_classifier,
                              wilson_interval)
from equipment.taxonomy import Taxonomy


def asset(aid, family="f1", site="s1"):
    return Asset(aid, family, site, date(2024, 1, 1), None, None)


def report(rid, aid, day, text):
    return Report(rid, aid, date.fromisoformat(day), text, "web")


def disposition(aid, event, effective, category):
    return Disposition(aid, event, date.fromisoformat(effective), category, "verified")


def build(assets, reports, dispositions):
    taxonomy = Taxonomy(DataBundle(ROOT / "data").taxonomy)
    identity = IdentityResolver(assets)
    classifications = {
        c.report: c
        for c in RuleBasedClassifier(taxonomy).classify_many(
            reports, {a.asset: a.family for a in assets})
    }
    events = EventBuilder(assets, reports, dispositions, identity, classifications).build()
    return taxonomy, events


class QualityEvaluationTest(unittest.TestCase):
    def setUp(self):
        # 两个型号各一台设备，各经历 修复前→修复后 两个核实事件
        self.assets = [asset("a1", "f1"), asset("a2", "f9")]
        self.reports = [
            report("r1", "a1", "2026-07-01", "异响需要复查"),
            report("r2", "a1", "2026-07-10", "异响需要复查"),
            report("r3", "a2", "2026-07-01", "座椅松动影响使用"),
            report("r4", "a2", "2026-07-10", "座椅松动影响使用"),
        ]
        self.dispositions = [
            disposition("a1", "e1", "2026-07-05", "c0"),
            disposition("a1", "e2", "2026-07-15", "c0"),
            disposition("a2", "e3", "2026-07-05", "c1"),
            disposition("a2", "e4", "2026-07-15", "c1"),
        ]
        self.taxonomy, self.events = build(self.assets, self.reports, self.dispositions)
        self.family_by_asset = {"a1": "f1", "a2": "f9"}

    def test_temporal_split_keeps_event_and_duplicates_together(self):
        split = temporal_event_split(self.events, date(2026, 7, 8))
        self.assertEqual({e.event_id for e in split.train_events},
                         {e.event_id for e in self.events if e.window_index == 0})
        split.assert_no_leakage()
        self.assertFalse(split.train_reports & split.test_reports)

    def test_leakage_guard_detects_overlap(self):
        from equipment.quality import EventSplit
        bad = EventSplit(
            train_events=tuple(self.events[:1]),
            test_events=tuple(self.events),
            train_reports=frozenset({"r1"}),
            test_reports=frozenset({"r1", "r2"}),
        )
        with self.assertRaises(AssertionError):
            bad.assert_no_leakage()

    def test_ground_truth_only_from_verified_events(self):
        truth = build_ground_truth(self.events)
        self.assertEqual(truth, {"r1": "c0", "r2": "c0", "r3": "c1", "r4": "c1"})

    def test_error_rate_measured_on_later_events(self):
        split = temporal_event_split(self.events, date(2026, 7, 8))
        classifier = train_scoped_classifier(self.taxonomy, list(split.train_events))
        test_reports = [r for r in self.reports if r.report in split.test_reports]
        score = score_classifier(classifier, test_reports,
                                 build_ground_truth(self.events), self.family_by_asset)
        self.assertEqual(score.scored, 2)
        self.assertEqual(score.wrong, 0)
        self.assertEqual(score.error_rate.point, 0.0)

    def test_held_out_family_is_abstention_scope_not_error_scope(self):
        covered, uncovered = family_holdout_split(self.events, {"f9"})
        self.assertTrue(all(e.family != "f9" for e in covered))
        classifier = RuleBasedClassifier(self.taxonomy, scoped_families={"f1"})
        holdout_reports = [r for e in uncovered for link in e.links
                           for r in self.reports if r.report == link.report]
        score = score_classifier(classifier, holdout_reports,
                                 build_ground_truth(self.events), self.family_by_asset)
        self.assertEqual(score.abstained, 2)
        self.assertEqual(score.scored, 0)
        self.assertEqual(score.abstain_rate.point, 1.0)

    def test_merge_score_pairwise_precision_recall(self):
        merge = score_merges(self.events)
        # 修复边界把前后报告切成独立事件，本数据无成对重复件
        self.assertEqual(merge.true_pairs, 0)
        self.assertEqual(merge.predicted_pairs, 0)
        self.assertEqual(merge.provisional_in_official_stats, 0)

    def test_merge_score_detects_within_window_duplicate_pairs(self):
        reports = self.reports + [
            report("r5", "a1", "2026-07-02", "异响需要复查", ),
            report("r6", "a1", "2026-07-03", "异响需要复查"),
        ]
        _, events = build(self.assets, reports, self.dispositions)
        merge = score_merges(events)
        # w00 窗口内 r1/r5/r6 三条同类别 → 3 对真值且全部正确预测
        self.assertEqual(merge.true_pairs, 3)
        self.assertEqual(merge.predicted_pairs, 3)
        self.assertEqual(merge.correct_pairs, 3)
        self.assertEqual(merge.merge_precision.point, 1.0)
        self.assertEqual(merge.merge_recall.point, 1.0)

    def test_wilson_interval_bounds(self):
        estimate = wilson_interval(3, 10)
        self.assertEqual(estimate.denominator, 10)
        self.assertGreaterEqual(estimate.ci_low, 0.0)
        self.assertLessEqual(estimate.ci_high, 1.0)
        empty = wilson_interval(0, 0)
        self.assertEqual(empty.point, 0.0)


if __name__ == "__main__":
    unittest.main()
