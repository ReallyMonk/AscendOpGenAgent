#!/usr/bin/env python3
"""Benchmark registry and configuration loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ServerInfo:
    name: str
    worker_url: str
    ssh_host: Optional[str]
    hardware: Optional[str]
    default_device: str
    client_id: str
    notes: Optional[str] = None


class ServerRegistry:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.default_server = data["default_server"]
        self.servers = data["servers"]

    @classmethod
    def from_file(cls, path: str | Path) -> "ServerRegistry":
        return cls(load_json(path))

    def get(self, name: Optional[str] = None) -> ServerInfo:
        server_name = name or self.default_server
        raw = self.servers[server_name]
        return ServerInfo(
            name=server_name,
            worker_url=raw["worker_url"],
            ssh_host=raw.get("ssh_host"),
            hardware=raw.get("hardware"),
            default_device=str(raw.get("default_device", "0")),
            client_id=raw.get("client_id", server_name),
            notes=raw.get("notes"),
        )


class RunnerTemplateRegistry:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.templates = data["templates"]

    @classmethod
    def from_file(cls, path: str | Path) -> "RunnerTemplateRegistry":
        return cls(load_json(path))

    def get(self, op_name: str) -> Dict[str, Any]:
        return self.templates[op_name]

    def has(self, op_name: str) -> bool:
        return op_name in self.templates


class BenchmarkRegistry:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.operators = data["operators"]
        self.benchmark_sets = data.get("benchmark_sets", {})

    @classmethod
    def from_file(cls, path: str | Path) -> "BenchmarkRegistry":
        return cls(load_json(path))

    def get_operator(self, op_name: str) -> Dict[str, Any]:
        return self.operators[op_name]

    def get_shape(self, op_name: str, shape_id: str) -> Dict[str, Any]:
        for shape in self.operators[op_name]["shapes"]:
            if shape["id"] == shape_id:
                return shape
        raise KeyError(f"shape_id={shape_id} not found for op={op_name}")

    def iter_operators(
        self,
        *,
        set_name: Optional[str] = None,
        layer: Optional[str] = None,
        role: Optional[str] = None,
        category: Optional[str] = None,
        op_name: Optional[str] = None,
    ) -> Iterable[tuple[str, Dict[str, Any]]]:
        if op_name:
            op = self.get_operator(op_name)
            if self._match(op_name, op, set_name=set_name, layer=layer, role=role, category=category):
                yield op_name, op
            return

        for name, op in self.operators.items():
            if self._match(name, op, set_name=set_name, layer=layer, role=role, category=category):
                yield name, op

    def _match(
        self,
        op_name: str,
        op: Dict[str, Any],
        *,
        set_name: Optional[str],
        layer: Optional[str],
        role: Optional[str],
        category: Optional[str],
    ) -> bool:
        if set_name:
            selected = self.benchmark_sets.get(set_name, [])
            if op_name not in selected:
                return False
        if layer and op.get("benchmark_layer") != layer:
            return False
        if role and op.get("benchmark_role") != role:
            return False
        if category and op.get("category") != category:
            return False
        return True
