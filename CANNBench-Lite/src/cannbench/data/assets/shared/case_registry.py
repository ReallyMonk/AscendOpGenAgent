"""Unified case registry for evaluation skills.

Provides CaseRecord and CaseRegistry for persisting case spec + seed to
cases.json, enabling seed-based replay without tensor file persistence.
"""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class CaseRecord:
    """Single test case record with enough info to regenerate tensors."""
    case_id: str
    seed: int
    generator: str
    status: str
    spec: dict


class CaseRegistry:
    """Registry of test cases for a single skill + operator run.

    Serializes to / deserializes from cases.json.
    """

    VERSION = 1

    def __init__(self, skill_name: str, op_name: str, config: dict | None = None):
        self.skill_name = skill_name
        self.op_name = op_name
        self.config = config or {}
        self.records: List[CaseRecord] = []

    def add(self, record: CaseRecord) -> None:
        self.records.append(record)

    def update_status(self, case_id: str, status: str) -> None:
        for r in self.records:
            if r.case_id == case_id:
                r.status = status
                return

    def save(self, output_path: Path) -> Path:
        """Write cases.json to output_path directory. Returns the file path."""
        os.makedirs(output_path, exist_ok=True)
        file_path = output_path / "cases.json"
        data = {
            "version": self.VERSION,
            "skill": self.skill_name,
            "op_name": self.op_name,
            "config": self.config,
            "cases": [asdict(r) for r in self.records],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return file_path

    @classmethod
    def load(cls, path: Path) -> "CaseRegistry":
        """Load CaseRegistry from a cases.json file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        reg = cls(
            skill_name=data["skill"],
            op_name=data["op_name"],
            config=data.get("config", {}),
        )
        for c in data["cases"]:
            reg.add(CaseRecord(
                case_id=c["case_id"],
                seed=c["seed"],
                generator=c["generator"],
                status=c["status"],
                spec=c["spec"],
            ))
        return reg

    def filter(
        self,
        status: Optional[List[str]] = None,
        case_ids: Optional[List[str]] = None,
    ) -> List[CaseRecord]:
        """Filter records by status and/or case_ids. Both filters are AND-ed."""
        result = self.records
        if status is not None:
            result = [r for r in result if r.status in status]
        if case_ids is not None:
            id_set = set(case_ids)
            result = [r for r in result if r.case_id in id_set]
        return result
