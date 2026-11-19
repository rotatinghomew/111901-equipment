"""端到端离线分析流水线。

只读取本地资料并产出结构化分析结果；不含任何远程设备操作，
也不会虚构现场复核结论（复核结果由外部输入）。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from .classify import RuleBasedClassifier
from .events import EventBuilder, recurrence_by_family
from .identity import IdentityResolver
from .io import DataBundle
from .quality import (build_ground_truth, family_holdout_split,
                      score_classifier, score_merges, temporal_event_split,
                      train_scoped_classifier)
from .review import CapacityLedger, ReviewPlanner
from .taxonomy import Taxonomy


class AnalysisPipeline:
    def __init__(self, bundle: DataBundle, *, cutoff: date,
                 held_out_families: set[str],
                 bookings: list[dict[str, object]] | None = None):
        self.bundle = bundle
        self.cutoff = cutoff
        self.held_out_families = held_out_families
        self.bookings = bookings or []
        self.taxonomy = Taxonomy(bundle.taxonomy)
        self.family_by_asset = {a.asset: a.family for a in bundle.assets}
        self.site_by_asset = {a.asset: a.site for a in bundle.assets}

    def run(self) -> dict[str, object]:
        # 1. 物理身份与迁移链
        identity = IdentityResolver(self.bundle.assets)

        # 2. 全量描述分类（带凭据）
        classifier = RuleBasedClassifier(self.taxonomy)
        classifications = {
            c.report: c
            for c in classifier.classify_many(self.bundle.reports, self.family_by_asset)
        }

        # 3. 事件聚合
        builder = EventBuilder(self.bundle.assets, self.bundle.reports,
                               self.bundle.dispositions, identity, classifications)
        events = builder.build()

        # 4. 质量评估：时间后移 + 型号留出，防泄漏
        split = temporal_event_split(events, self.cutoff)
        split.assert_no_leakage()
        ground_truth = build_ground_truth(events)
        scoped = train_scoped_classifier(self.taxonomy, list(split.train_events))
        test_reports = [r for r in self.bundle.reports if r.report in split.test_reports]
        temporal_score = score_classifier(scoped, test_reports, ground_truth,
                                          self.family_by_asset)
        covered_events, uncovered_events = family_holdout_split(events, self.held_out_families)
        # 型号留出是独立口径：分类器完全未见过留出型号，必须在其上弃权。
        holdout_classifier = RuleBasedClassifier(
            self.taxonomy,
            scoped_families={a.family for a in self.bundle.assets} - self.held_out_families,
        )
        holdout_reports = [r for e in uncovered_events for link in e.links
                           for r in self.bundle.reports if r.report == link.report]
        holdout_score = score_classifier(holdout_classifier, holdout_reports, ground_truth,
                                         self.family_by_asset)
        merge_score = score_merges(events)

        # 5. 抽核分配
        safety_event_ids = {
            event.event_id for event in events
            if any(classifications[link.report].safety_suspected for link in event.links)
        }
        planner = ReviewPlanner(self.bundle.sampling, self.taxonomy.skill_for)
        units = planner.build_units(
            events, safety_event_ids=safety_event_ids,
            unseen_families=self.held_out_families,
            site_by_asset=self.site_by_asset,
        )
        ledger = CapacityLedger(self.bundle.routes, self.bookings)
        allocation = planner.allocate(units, ledger)

        return {
            "physical_units": [
                {
                    "unit_id": u.unit_id, "assets": list(u.assets), "family": u.family,
                    "sites": list(u.sites), "confirmed": u.confirmed, "note": u.note,
                }
                for u in identity.units
            ],
            "ambiguous_migration_links": identity.ambiguous_links(),
            "colocated_same_family_pairs": identity.colocated_same_family_pairs(),
            "classification_examples": [
                {
                    "report": c.report, "version": c.taxonomy_version,
                    "category": c.category, "chase": c.chase,
                    "uncertain": c.uncertain, "impact": c.impact,
                    "safety_suspected": c.safety_suspected,
                    "evidence": list(c.evidence),
                }
                for c in list(classifications.values())[:6]
            ],
            "events": [self._event_view(event) for event in events],
            "recurrence_by_family": recurrence_by_family(events),
            "quality": {
                "cutoff": self.cutoff.isoformat(),
                "train_events": len(split.train_events),
                "test_events": len(split.test_events),
                "temporal_error_rate": temporal_score.error_rate.as_dict(),
                "temporal_mismatches": list(temporal_score.mismatches),
                "family_holdout": {
                    "families": sorted(self.held_out_families),
                    "abstain_rate": holdout_score.abstain_rate.as_dict(),
                    "scored": holdout_score.scored,
                    "abstained": holdout_score.abstained,
                },
                "merge_quality": {
                    "true_pairs": merge_score.true_pairs,
                    "predicted_pairs": merge_score.predicted_pairs,
                    "correct_pairs": merge_score.correct_pairs,
                    "precision": merge_score.merge_precision.as_dict(),
                    "recall": merge_score.merge_recall.as_dict(),
                    "provisional_links": merge_score.provisional_links,
                    "provisional_in_official_stats": merge_score.provisional_in_official_stats,
                },
            },
            "review_allocation": {
                "unit_count": len(units),
                "assignments": [a.__dict__ for a in allocation.assignments],
                "escalated": [a.__dict__ for a in allocation.escalated],
                "deferred": [a.__dict__ for a in allocation.deferred],
                "capacity_remaining": {f"{k[0]}|{k[1]}": v
                                       for k, v in allocation.capacity_remaining.items()},
                "rotation_design": allocation.rotation_design,
            },
        }

    @staticmethod
    def _event_view(event) -> dict[str, object]:
        return {
            "event_id": event.event_id,
            "unit_id": event.unit_id,
            "family": event.family,
            "opened_at": event.opened_at.isoformat(),
            "closed_at": event.closed_at.isoformat() if event.closed_at else None,
            "status": event.status,
            "provisional": event.provisional,
            "verified_category": event.verified_category,
            "chase_count": event.chase_count,
            "links": [link.__dict__ | {"submitted_at": link.submitted_at.isoformat()}
                      for link in event.links],
            "pending_dispositions": [
                {"asset": d.asset, "event": d.event, "status": d.status}
                for d in event.pending_dispositions
            ],
        }


def run_default(data_dir: str | Path, *, cutoff: date | None = None,
                held_out_families: set[str] | None = None,
                bookings: list[dict[str, object]] | None = None) -> dict[str, object]:
    bundle = DataBundle(data_dir)
    pipeline = AnalysisPipeline(
        bundle,
        cutoff=cutoff or date(2026, 7, 15),
        held_out_families=held_out_families or {"f5"},
        bookings=bookings,
    )
    return pipeline.run()
