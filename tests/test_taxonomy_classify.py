import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from equipment.classify import RuleBasedClassifier
from equipment.io import DataBundle
from equipment.models import Report
from equipment.taxonomy import Taxonomy


class TaxonomyVersionTest(unittest.TestCase):
    def setUp(self):
        self.tax = Taxonomy(DataBundle(ROOT / "data").taxonomy)

    def test_version_window_boundary(self):
        self.assertEqual(self.tax.version_for(date(2026, 7, 19)), "v1")
        self.assertEqual(self.tax.version_for(date(2026, 7, 20)), "v2")

    def test_old_label_is_explained_not_resplit(self):
        explanation = self.tax.explain_label("c1", "v1")
        self.assertEqual(explanation["mapping"], "retained")
        self.assertIn("不硬拆", explanation["lineage_note"])
        # v2 的细类必须有父类，老标签只能回到父类而非被猜成细类
        self.assertEqual(self.tax.category("c1.1").parent, "c1")

    def test_v2_uses_subcategory_only_with_evidence(self):
        # 仅"松动"无部位证据 → 父类 c1，不硬拆成 c1.1
        plain = RuleBasedClassifier(self.tax).classify(
            Report("x1", "a01", date(2026, 7, 21), "松动需要处理", "web"), "f1")
        self.assertEqual(plain.category, "c1")
        seat = RuleBasedClassifier(self.tax).classify(
            Report("x2", "a01", date(2026, 7, 21), "座椅松动影响使用", "web"), "f1")
        self.assertEqual(seat.category, "c1.1")

    def test_evidence_and_uncertainty_and_impact(self):
        result = RuleBasedClassifier(self.tax).classify(
            Report("x3", "a01", date(2026, 7, 3), "疑似保护罩故障", "operator"), "f1")
        self.assertEqual(result.category, "c2")
        self.assertTrue(result.uncertain)
        self.assertTrue(result.safety_suspected)
        self.assertTrue(any("不确定标记" in e for e in result.evidence))
        self.assertTrue(any("命中规则" in e for e in result.evidence))

    def test_chase_is_not_a_new_fault_category(self):
        result = RuleBasedClassifier(self.tax).classify(
            Report("x4", "a01", date(2026, 7, 4), "持续催问是否修好", "inspection"), "f1")
        self.assertTrue(result.chase)
        self.assertIsNone(result.category)
        self.assertEqual(result.impact, "none")

    def test_abstain_when_no_rule_matches(self):
        result = RuleBasedClassifier(self.tax).classify(
            Report("x5", "a01", date(2026, 7, 4), "颜色褪色不好看", "web"), "f1")
        self.assertIsNone(result.category)
        self.assertTrue(any("未判定原因=no_rule_matched" in e for e in result.evidence))


if __name__ == "__main__":
    unittest.main()
