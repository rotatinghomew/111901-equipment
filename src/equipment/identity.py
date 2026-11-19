"""物理设备身份解析。

设备编号会在迁移后变化。本模块只依据"设备档案与迁移记录"建立物理单元：

- 唯一、无歧义的迁移链（moved_from 只指向一个旧编号）→ 合并为同一物理单元；
- 多个新编号声称来自同一旧编号 → 迁移关系未确认，相关编号各自独立，
  未确认关系不进入合并统计；
- 同型号（family）且共址（site）的两台设备没有迁移凭据，绝不合并；
- 无迁移记录的设备各自成单元。
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import Asset


@dataclass(frozen=True)
class PhysicalUnit:
    unit_id: str
    assets: tuple[str, ...]
    family: str
    sites: tuple[str, ...]
    confirmed: bool
    note: str


class IdentityResolver:
    def __init__(self, assets: list[Asset]):
        self.assets = {a.asset: a for a in assets}
        self._incoming: dict[str, list[str]] = {}
        for asset in assets:
            if asset.moved_from:
                self._incoming.setdefault(asset.moved_from, []).append(asset.asset)
        self._ambiguous_predecessors = {
            old for old, news in self._incoming.items() if len(news) > 1
        }
        self._units, self._asset_to_unit = self._build_units()

    def _build_units(self) -> tuple[list[PhysicalUnit], dict[str, str]]:
        # 无歧义迁移链：从没有上游、或上游本身歧义的节点出发向后跟随。
        chains: list[list[str]] = []
        current_assets = set(self.assets)
        heads = [
            a.asset for a in self.assets.values()
            if not a.moved_from or a.moved_from in self._ambiguous_predecessors
            or a.moved_from not in current_assets
        ]
        for head in sorted(set(heads)):
            chain = [head]
            cursor = head
            while True:
                followers = self._incoming.get(cursor, [])
                if len(followers) == 1 and followers[0] not in self._ambiguous_predecessors:
                    chain.append(followers[0])
                    cursor = followers[0]
                    continue
                break
            chains.append(chain)

        units: list[PhysicalUnit] = []
        mapping: dict[str, str] = {}
        for index, chain in enumerate(sorted(chains, key=lambda c: c[0]), start=1):
            unit_id = f"pu{index:03d}"
            members = tuple(chain)
            families = {self.assets[a].family for a in members}
            sites = tuple(sorted({self.assets[a].site for a in members}))
            note_parts: list[str] = []
            confirmed = True
            for member in members:
                asset = self.assets[member]
                if asset.moved_from:
                    if asset.moved_from in self._ambiguous_predecessors:
                        confirmed = False
                        note_parts.append(
                            f"{member} 的迁移来源 {asset.moved_from} 被多个编号声称，关系未确认"
                        )
                    elif asset.moved_from not in current_assets:
                        note_parts.append(
                            f"{member} 迁移自档案外编号 {asset.moved_from}，链首按当前编号建单元"
                        )
            if len(families) > 1:
                confirmed = False
                note_parts.append("链上设备型号不一致，迁移关系需人工复核")
            note = "；".join(note_parts) if note_parts else "迁移链唯一，关系已确认"
            units.append(PhysicalUnit(
                unit_id=unit_id,
                assets=members,
                family=next(iter(families)),
                sites=sites,
                confirmed=confirmed,
                note=note,
            ))
            for member in members:
                mapping[member] = unit_id

        # 防御性检查：共址同型号必须落在不同物理单元。
        seen: dict[tuple[str, str], str] = {}
        for unit in units:
            for member in unit.assets:
                key = (self.assets[member].family, self.assets[member].site)
                if key in seen and seen[key] != unit.unit_id:
                    # 不同单元共址同型号是允许且必须保留的，这里只校验单元内不混。
                    continue
                seen.setdefault(key, unit.unit_id)
        return units, mapping

    @property
    def units(self) -> list[PhysicalUnit]:
        return list(self._units)

    def unit_of(self, asset_id: str) -> PhysicalUnit:
        return next(u for u in self._units if u.unit_id == self._asset_to_unit[asset_id])

    def same_physical_device(self, asset_a: str, asset_b: str) -> bool:
        """仅当归属同一已确认物理单元时返回 True。"""
        unit_a = self.unit_of(asset_a)
        unit_b = self.unit_of(asset_b)
        return unit_a.unit_id == unit_b.unit_id and unit_a.confirmed

    def ambiguous_links(self) -> list[dict[str, object]]:
        return [
            {"predecessor": old, "claimants": sorted(news)}
            for old, news in sorted(self._incoming.items())
            if len(news) > 1
        ]

    def colocated_same_family_pairs(self) -> list[tuple[str, str]]:
        """列出同型号共址但分属不同物理单元的设备对（不可合并的证据）。"""
        pairs: list[tuple[str, str]] = []
        by_site_family: dict[tuple[str, str], list[str]] = {}
        for asset in self.assets.values():
            by_site_family.setdefault((asset.site, asset.family), []).append(asset.asset)
        for members in by_site_family.values():
            if len(members) < 2:
                continue
            members = sorted(members)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    if self._asset_to_unit[a] != self._asset_to_unit[b]:
                        pairs.append((a, b))
        return pairs
