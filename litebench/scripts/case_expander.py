#!/usr/bin/env python3
"""Expand registry shapes into concrete executable input specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ExpandedInput:
    name: str
    shape: List[int]
    spec: Dict[str, Any]


@dataclass(frozen=True)
class ExpandedCase:
    op_name: str
    shape_id: str
    benchmark_layer: str
    benchmark_role: str
    category: str
    runner_type: str
    api: str | None
    attrs: Dict[str, Any]
    inputs: List[ExpandedInput]
    raw_shape: Dict[str, Any]
    raw_operator: Dict[str, Any]
    raw_template: Dict[str, Any]


def _resolve_dim(value: Any, params: Dict[str, Any]) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        if value not in params:
            raise KeyError(f"shape token {value!r} missing from params {sorted(params)}")
        return int(params[value])
    raise TypeError(f"unsupported dim type: {type(value).__name__}")


def _resolve_value(value: Any, params: Dict[str, Any]) -> Any:
    if isinstance(value, list):
        return [_resolve_value(item, params) for item in value]
    if isinstance(value, dict):
        resolved: Dict[str, Any] = {}
        for key, item in value.items():
            if key.endswith("_from_param"):
                target_key = key[: -len("_from_param")]
                resolved[target_key] = _resolve_value(params[item], params)
            else:
                resolved[key] = _resolve_value(item, params)
        return resolved
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        if value in params:
            return params[value]
    return value


def expand_case(op_name: str, operator: Dict[str, Any], shape: Dict[str, Any], template: Dict[str, Any]) -> ExpandedCase:
    params = shape["params"]
    inputs: List[ExpandedInput] = []
    for input_name, spec in template.get("inputs", {}).items():
        raw_shape = _resolve_value(spec["shape"], params)
        resolved = [_resolve_dim(dim, params) for dim in raw_shape]
        resolved_spec = _resolve_value(spec, params)
        inputs.append(ExpandedInput(name=input_name, shape=resolved, spec=resolved_spec))

    attrs = _resolve_value(template.get("attrs", {}), params)

    return ExpandedCase(
        op_name=op_name,
        shape_id=shape["id"],
        benchmark_layer=operator["benchmark_layer"],
        benchmark_role=operator["benchmark_role"],
        category=operator["category"],
        runner_type=template.get("runner_type", "unknown"),
        api=template.get("api"),
        attrs=attrs,
        inputs=inputs,
        raw_shape=shape,
        raw_operator=operator,
        raw_template=template,
    )
