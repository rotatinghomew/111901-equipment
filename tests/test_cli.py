import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from equipment.cli import main, parse_bookings


class CliTest(unittest.TestCase):
    def test_parse_bookings(self):
        self.assertEqual(
            parse_bookings(["s1:mechanical:2", "s0:inspection:1"]),
            [{"site": "s1", "skill": "mechanical", "slots": 2},
             {"site": "s0", "skill": "inspection", "slots": 1}],
        )

    def test_main_writes_structured_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result.json"
            code = main(["--data-dir", str(ROOT / "data"),
                         "--cutoff", "2026-07-15", "--holdout", "f5",
                         "--out", str(out)])
            self.assertEqual(code, 0)
            result = json.loads(out.read_text(encoding="utf-8"))
            for key in ("physical_units", "events", "quality", "review_allocation",
                        "recurrence_by_family", "ambiguous_migration_links"):
                self.assertIn(key, result)
            self.assertGreater(len(result["events"]), 0)


if __name__ == "__main__":
    unittest.main()
