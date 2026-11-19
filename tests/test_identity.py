import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from equipment.identity import IdentityResolver
from equipment.io import DataBundle
from equipment.models import Asset


def make_asset(asset, family="f1", site="s1", moved_from=None, moved_at=None):
    return Asset(asset, family, site, date(2024, 1, 1), moved_from, moved_at)


class IdentityResolutionTest(unittest.TestCase):
    def test_unambiguous_chain_merges_into_one_physical_unit(self):
        assets = [
            make_asset("n1", moved_from="old9", moved_at=date(2026, 6, 1)),
            make_asset("n2", moved_from="n1", moved_at=date(2026, 6, 2)),
        ]
        resolver = IdentityResolver(assets)
        self.assertEqual(len(resolver.units), 1)
        self.assertTrue(resolver.same_physical_device("n1", "n2"))
        self.assertTrue(resolver.unit_of("n1").confirmed)

    def test_ambiguous_migration_is_not_merged(self):
        assets = [
            make_asset("n1", moved_from="oldX", moved_at=date(2026, 6, 1)),
            make_asset("n2", moved_from="oldX", moved_at=date(2026, 6, 2)),
        ]
        resolver = IdentityResolver(assets)
        self.assertEqual(len(resolver.units), 2)
        self.assertFalse(resolver.same_physical_device("n1", "n2"))
        self.assertEqual(resolver.ambiguous_links(),
                         [{"predecessor": "oldX", "claimants": ["n1", "n2"]}])
        self.assertFalse(all(u.confirmed for u in resolver.units))

    def test_colocated_same_family_never_merges(self):
        assets = [make_asset("a", site="s1"), make_asset("b", site="s1")]
        resolver = IdentityResolver(assets)
        self.assertEqual(len(resolver.units), 2)
        self.assertIn(("a", "b"), resolver.colocated_same_family_pairs())

    def test_fixture_contains_one_ambiguous_predecessor(self):
        bundle_assets = DataBundle(ROOT / "data").assets
        resolver = IdentityResolver(bundle_assets)
        ambiguous = resolver.ambiguous_links()
        self.assertEqual(ambiguous, [{"predecessor": "old3", "claimants": ["a07", "a35"]}])
        self.assertFalse(resolver.same_physical_device("a07", "a35"))
        # 档案外旧编号不阻断建单元
        self.assertTrue(all(u.assets for u in resolver.units))


if __name__ == "__main__":
    unittest.main()
