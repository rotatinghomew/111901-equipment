"""命令行入口：python -m equipment.cli --data-dir data --out result.json

只做离线读取与分析输出，不执行任何远程设备操作。
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .pipeline import AnalysisPipeline
from .io import DataBundle


def parse_bookings(values: list[str]) -> list[dict[str, object]]:
    bookings: list[dict[str, object]] = []
    for value in values:
        site, skill, slots = value.split(":")
        bookings.append({"site": site, "skill": skill, "slots": int(slots)})
    return bookings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="器材报障聚合分析与抽核分配（离线）")
    parser.add_argument("--data-dir", default="data", help="CSV 资料目录")
    parser.add_argument("--policy-dir", default=None, help="策略 JSON 目录")
    parser.add_argument("--cutoff", default="2026-07-15",
                        help="时间后移切片日期（含当日起为验收侧）")
    parser.add_argument("--holdout", default="f5",
                        help="型号留出，逗号分隔，如 f5,f6")
    parser.add_argument("--booking", action="append", default=[],
                        metavar="SITE:SKILL:SLOTS",
                        help="已约人员占用名额，可重复传入")
    parser.add_argument("--out", default=None, help="结果 JSON 输出路径；缺省打印到标准输出")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = DataBundle(args.data_dir, args.policy_dir)
    pipeline = AnalysisPipeline(
        bundle,
        cutoff=date.fromisoformat(args.cutoff),
        held_out_families={f.strip() for f in args.holdout.split(",") if f.strip()},
        bookings=parse_bookings(args.booking),
    )
    result = pipeline.run()
    payload = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
