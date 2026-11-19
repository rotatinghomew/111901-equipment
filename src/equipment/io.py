"""CSV/JSON 资料加载。

空字段统一转为 None；日期字段按本地日期解析。
加载器只做读取与结构校验，不做任何业务推断。
"""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from .models import Asset, Disposition, Report, Route


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _parse_date(value: str | None) -> date | None:
    value = _blank_to_none(value)
    if value is None:
        return None
    return date.fromisoformat(value)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_assets(path: str | Path) -> list[Asset]:
    rows = _read_rows(Path(path))
    assets = [
        Asset(
            asset=row["asset"],
            family=row["family"],
            site=row["site"],
            commissioned=_parse_date(row.get("commissioned")),
            moved_from=_blank_to_none(row.get("moved_from")),
            moved_at=_parse_date(row.get("moved_at")),
        )
        for row in rows
    ]
    keys = [a.asset for a in assets]
    if len(keys) != len(set(keys)):
        raise ValueError("assets.csv 存在重复设备编号")
    return assets


def load_reports(path: str | Path) -> list[Report]:
    rows = _read_rows(Path(path))
    reports = [
        Report(
            report=row["report"],
            asset=row["asset"],
            submitted_at=_parse_date(row["submitted_at"]),
            text=row["text"],
            source=row["source"],
        )
        for row in rows
    ]
    keys = [r.report for r in reports]
    if len(keys) != len(set(keys)):
        raise ValueError("reports.csv 存在重复报障单号")
    return reports


def load_dispositions(path: str | Path) -> list[Disposition]:
    rows = _read_rows(Path(path))
    dispositions = [
        Disposition(
            asset=row["asset"],
            event=row["event"],
            repair_effective=_parse_date(row.get("repair_effective")),
            category=_blank_to_none(row.get("category")) or "",
            status=row["status"],
        )
        for row in rows
    ]
    # (asset, event) 是处置事件的复合键，event 列本身不全局唯一。
    keys = [(d.asset, d.event) for d in dispositions]
    if len(keys) != len(set(keys)):
        raise ValueError("dispositions.csv 存在重复 (asset, event) 组合")
    return dispositions


def load_routes(path: str | Path) -> list[Route]:
    rows = _read_rows(Path(path))
    routes = [
        Route(
            site=row["site"],
            route=row["route"],
            skill=row["skill"],
            weekly_slots=int(row["weekly_slots"]),
        )
        for row in rows
    ]
    keys = [r.route for r in routes]
    if len(keys) != len(set(keys)):
        raise ValueError("routes.csv 存在重复路线编号")
    return routes


def load_policy(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"策略文件须为 JSON 对象: {path}")
    return value


class DataBundle:
    """一次性加载全部离线资料的便捷容器。"""

    def __init__(self, data_dir: str | Path, policy_dir: str | Path | None = None):
        data_dir = Path(data_dir)
        policy_dir = Path(policy_dir) if policy_dir else data_dir / "policy"
        self.assets = load_assets(data_dir / "assets.csv")
        self.reports = load_reports(data_dir / "reports.csv")
        self.dispositions = load_dispositions(data_dir / "dispositions.csv")
        self.routes = load_routes(data_dir / "routes.csv")
        self.taxonomy = load_policy(policy_dir / "taxonomy.json")
        self.sampling = load_policy(policy_dir / "sampling.json")
        self._check_referential_integrity()

    def _check_referential_integrity(self) -> None:
        asset_ids = {a.asset for a in self.assets}
        unknown_report_assets = {r.asset for r in self.reports} - asset_ids
        if unknown_report_assets:
            raise ValueError(f"报障引用了未知设备: {sorted(unknown_report_assets)}")
        unknown_disp_assets = {d.asset for d in self.dispositions} - asset_ids
        if unknown_disp_assets:
            raise ValueError(f"处置记录引用了未知设备: {sorted(unknown_disp_assets)}")
        route_sites = {r.site for r in self.routes}
        unknown_sites = {a.site for a in self.assets} - route_sites
        if unknown_sites:
            raise ValueError(f"设备所在地点无巡视路线: {sorted(unknown_sites)}")
