#!/usr/bin/env python3
"""
Knowledge Card MCP Server

Structured MCP server for fixed-schema knowledge cards.

Recommended workflow:
1. search_knowledge_card_summaries(...) for discovery
2. get_knowledge_card_by_id(case_id) for full read
3. create_knowledge_card_from_diff(...) as the preferred write path
4. update_knowledge_card_effect(...) for profiling/effect backfill
5. save_knowledge_card(...) only when a full valid card already exists
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    raise ImportError(
        "Please install the MCP SDK: pip install modelcontextprotocol"
    ) from e

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except ImportError as e:
    raise ImportError("Please install jsonschema: pip install jsonschema") from e


# ------------------------------------------------------------------------------
# Logging: IMPORTANT -> write logs to stderr, not stdout
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("knowledge-card-server")


# ------------------------------------------------------------------------------
# Paths / Config
#
# Resolution order (highest priority first):
#   1. KNOWLEDGE_CARDS_DIR       env var -> root of all data files
#   2. KNOWLEDGE_CARDS_SCHEMA    env var -> explicit schema file path
#   3. KNOWLEDGE_CARDS_BASE_DIR  env var -> project root (where schema and
#                                            other project files may live)
#   4. <KNOWLEDGE_CARDS_DIR>/thoughts_case_schema.json
#   5. <BASE_DIR>/thoughts_case_schema.json  (project-root fallback)
#   6. <package_dir>/schemas/thoughts_case_schema.json  (bundled fallback)
#
# All paths are overridable, so this module is fully standalone and can be
# pointed at any data directory in any project.
# ------------------------------------------------------------------------------

_PACKAGE_DIR = Path(__file__).resolve().parent
# Project root = parent of the package dir. With a src/ layout this is
# two levels up (src/knowledge_card -> src/ -> <project>), with a flat
# layout it is one level up. Default to the src-layout assumption; the
# KNOWLEDGE_CARDS_BASE_DIR env var can override.
PACKAGED_SCHEMA = _PACKAGE_DIR / "schemas" / "thoughts_case_schema.json"

BASE_DIR = Path(
    os.environ.get(
        "KNOWLEDGE_CARDS_BASE_DIR",
        str(_PACKAGE_DIR.parent.parent),
    )
).expanduser().resolve()

RECORDS_DIR = Path(
    os.environ.get("KNOWLEDGE_CARDS_DIR", str(BASE_DIR / "records"))
).expanduser().resolve()
CASES_DIR = RECORDS_DIR / "cases"
INDEX_FILE = RECORDS_DIR / "index.json"


def _resolve_schema_file() -> Path:
    """Pick the first existing schema file from the resolution order."""
    env_schema = os.environ.get("KNOWLEDGE_CARDS_SCHEMA")
    if env_schema:
        path = Path(env_schema).expanduser().resolve()
        if path.is_file():
            return path
    for candidate in (RECORDS_DIR / "thoughts_case_schema.json",
                      BASE_DIR / "thoughts_case_schema.json",
                      PACKAGED_SCHEMA):
        if candidate.is_file():
            return candidate
    # Fall back to project-root path so load_schema() raises a clear
    # FileNotFoundError at the actual missing location.
    return BASE_DIR / "thoughts_case_schema.json"


SCHEMA_FILE = _resolve_schema_file()

CASES_DIR.mkdir(parents=True, exist_ok=True)

CASE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

mcp = FastMCP("knowledge_cards")


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def success(data: Any = None, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"success": True}
    if data is not None:
        payload["data"] = data
    payload.update(extra)
    return payload


def failure(error: str, message: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "success": False,
        "error": error,
        "message": message,
    }
    payload.update(extra)
    return payload


def atomic_json_dump(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as tmp:
            json.dump(data, tmp, indent=2, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_name = tmp.name
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def validate_case_id(case_id: str) -> Tuple[bool, Optional[str]]:
    if not case_id:
        return False, "case_id is empty"
    if not CASE_ID_RE.fullmatch(case_id):
        return False, "case_id contains invalid characters; allowed: A-Z a-z 0-9 . _ -"
    return True, None


def build_case_id(kernel_name: str, session_id: str, step_index: int) -> str:
    parts = session_id.split("_")
    if len(parts) >= 4 and parts[0] == "sess":
        date_part = parts[-2]
        seq_part = parts[-1]
        return f"case_{kernel_name}_sess_{date_part}_{seq_part}_{step_index}"
    return f"case_{kernel_name}_{session_id}_{step_index}"


def infer_category(kernel_name: str) -> str:
    name_lower = kernel_name.lower()

    if "relu" in name_lower or "gelu" in name_lower or "swiglu" in name_lower:
        return "activation"
    if "layernorm" in name_lower or "layer_norm" in name_lower or "rmsnorm" in name_lower or "batchnorm" in name_lower:
        return "normalization"
    if "matmul" in name_lower or "linear" in name_lower or "gemm" in name_lower:
        return "matmul"
    if "conv" in name_lower:
        return "conv"
    if "softmax" in name_lower or "attention" in name_lower:
        return "attention"
    if "reduce" in name_lower or "sum" in name_lower or "max" in name_lower or "mean" in name_lower:
        return "reduction"
    return "elementwise"


def generate_thought_title(thought_id: str, kernel_name: str) -> str:
    if thought_id == "baseline":
        return f"{kernel_name} 基线测量"
    return f"{kernel_name} {thought_id.replace('_', ' ').title()}"


def map_category_to_opt_layer(change_category: str) -> str:
    mapping = {
        "memory": "tiling",
        "parallel": "core_partition",
        "compute": "instruction",
        "control": "pipeline",
    }
    return mapping.get(change_category.lower(), "tiling")


@lru_cache(maxsize=1)
def load_schema() -> Dict[str, Any]:
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_FILE}")
    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_validator() -> Draft202012Validator:
    schema = load_schema()
    return Draft202012Validator(schema)


def validate_knowledge_card_object(card: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    try:
        validator = get_validator()
        errors = sorted(validator.iter_errors(card), key=lambda e: list(e.path))
        if not errors:
            return True, None

        first = errors[0]
        path = ".".join(str(x) for x in first.path) if first.path else "<root>"
        return False, f"{path}: {first.message}"
    except jsonschema.SchemaError as e:
        return False, f"Invalid schema: {e.message}"


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index() -> Dict[str, Any]:
    if INDEX_FILE.exists():
        try:
            return load_json(INDEX_FILE)
        except Exception as e:
            logger.warning("Failed to load index.json, rebuilding empty index: %s", e)
    return {"cases": {}, "last_updated": None}


def save_index(index: Dict[str, Any]) -> None:
    index["last_updated"] = utc_now_iso()
    atomic_json_dump(index, INDEX_FILE)


def extract_card_summary(card: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": card.get("case_id", "unknown"),
        "kernel_name": card.get("kernel", {}).get("name", "unknown"),
        "category": card.get("kernel", {}).get("category", "unknown"),
        "compute_type": card.get("kernel", {}).get("compute_type", "unknown"),
        "effect_label": card.get("effect", {}).get("label", "unknown"),
        "step_improvement": card.get("effect", {}).get("step_improvement", None),
        "session_id": card.get("chain", {}).get("session_id", "unknown"),
        "step_index": card.get("chain", {}).get("step_index", None),
        "thought_id": card.get("thought", {}).get("id", "unknown"),
        "thought_title": card.get("thought", {}).get("title", ""),
        "target_bottleneck": card.get("thought", {}).get("target_bottleneck", "unknown"),
        "created_at": card.get("runtime", {}).get("created_at", None),
    }


def update_index_with_card(card: Dict[str, Any]) -> None:
    index = load_index()
    case_id = card["case_id"]
    index["cases"][case_id] = extract_card_summary(card)
    save_index(index)


def remove_from_index(case_id: str) -> None:
    index = load_index()
    if case_id in index.get("cases", {}):
        del index["cases"][case_id]
        save_index(index)


def card_path(case_id: str) -> Path:
    return CASES_DIR / f"{case_id}.json"


def load_card_by_id(case_id: str) -> Dict[str, Any]:
    return load_json(card_path(case_id))


def maybe_get(d: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    if not d:
        return default
    return d.get(key, default)


# ------------------------------------------------------------------------------
# Core MCP tools
# ------------------------------------------------------------------------------

@mcp.tool()
def search_knowledge_card_summaries(
    kernel_name: Optional[str] = None,
    category: Optional[str] = None,
    compute_type: Optional[str] = None,
    effect_label: Optional[str] = None,
    session_id: Optional[str] = None,
    thought_id: Optional[str] = None,
    target_bottleneck: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Search knowledge cards using indexed summary fields.

    Use this first for discovery when the exact case_id is not known.
    After finding candidates, call get_knowledge_card_by_id(case_id) to read
    the full card.

    All filters are exact-match filters. If no filters are provided, this returns
    recent summaries up to the given limit.
    """
    index = load_index()
    results: List[Dict[str, Any]] = []

    for summary in index.get("cases", {}).values():
        if kernel_name and summary.get("kernel_name") != kernel_name:
            continue
        if category and summary.get("category") != category:
            continue
        if compute_type and summary.get("compute_type") != compute_type:
            continue
        if effect_label and summary.get("effect_label") != effect_label:
            continue
        if session_id and summary.get("session_id") != session_id:
            continue
        if thought_id and summary.get("thought_id") != thought_id:
            continue
        if target_bottleneck and summary.get("target_bottleneck") != target_bottleneck:
            continue
        results.append(summary)

    # Prefer latest cards if created_at exists
    results.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
    results = results[: max(1, min(limit, 200))]

    return success(results, count=len(results))


@mcp.tool()
def get_knowledge_card_by_id(case_id: str) -> Dict[str, Any]:
    """
    Read the full knowledge card for one specific case_id.

    Use this when:
    - the exact case_id is already known
    - a previous search result returned candidate case_ids and full details are needed
    """
    ok, err = validate_case_id(case_id)
    if not ok:
        return failure("invalid_case_id", err or "Invalid case_id", case_id=case_id)

    path = card_path(case_id)
    if not path.exists():
        return failure("not_found", f"Knowledge card '{case_id}' not found", case_id=case_id)

    try:
        return success(load_card_by_id(case_id), case_id=case_id)
    except Exception as e:
        logger.exception("Failed to read card %s", case_id)
        return failure("read_failed", str(e), case_id=case_id)


@mcp.tool()
def save_knowledge_card(card: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
    """
    Save a complete knowledge card that already matches the required schema.

    Use this only when a full valid card object is already available.
    Prefer create_knowledge_card_from_diff(...) when the source input is a code diff,
    profiling result, or partially structured optimization record.
    """
    if not isinstance(card, dict):
        return failure("invalid_input", "card must be a dictionary")

    case_id = card.get("case_id")
    ok, err = validate_case_id(case_id)
    if not ok:
        return failure("invalid_case_id", err or "Invalid case_id", case_id=case_id)

    is_valid, error_msg = validate_knowledge_card_object(card)
    if not is_valid:
        return failure(
            "schema_validation_failed",
            error_msg or "Schema validation failed",
            case_id=case_id,
        )

    path = card_path(case_id)
    if path.exists() and not overwrite:
        return failure(
            "already_exists",
            f"Knowledge card '{case_id}' already exists; set overwrite=True to replace it",
            case_id=case_id,
        )

    try:
        atomic_json_dump(card, path)
        update_index_with_card(card)
        return success(
            {
                "case_id": case_id,
                "path": str(path),
                "overwritten": path.exists(),
            },
            case_id=case_id,
        )
    except Exception as e:
        logger.exception("Failed to save card %s", case_id)
        return failure("write_failed", str(e), case_id=case_id)


@mcp.tool()
def validate_knowledge_card(card: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a knowledge card against the schema without saving it.
    """
    if not isinstance(card, dict):
        return failure("invalid_input", "card must be a dictionary")

    is_valid, error_msg = validate_knowledge_card_object(card)
    if is_valid:
        return success({"valid": True, "message": "Knowledge card is valid"})
    return failure(
        "schema_validation_failed",
        error_msg or "Schema validation failed",
        valid=False,
    )


@mcp.tool()
def get_knowledge_card_schema() -> Dict[str, Any]:
    """
    Return the JSON schema used for knowledge card validation.

    Use this sparingly because it can be token-heavy.
    """
    try:
        return success(load_schema())
    except Exception as e:
        logger.exception("Failed to load schema")
        return failure("schema_load_failed", str(e))


@mcp.tool()
def update_knowledge_card_effect(
    case_id: str,
    effect_label: str,
    confidence: float,
    step_improvement: float,
    thought_match: Optional[str] = None,
    metric_changes: Optional[Dict[str, Any]] = None,
    overwrite_runtime_created_at: bool = False,
) -> Dict[str, Any]:
    """
    Update only the effect section of an existing knowledge card.

    Use this when the card already exists and new profiling/evaluation results
    need to be written back, without reconstructing the full card.
    """
    ok, err = validate_case_id(case_id)
    if not ok:
        return failure("invalid_case_id", err or "Invalid case_id", case_id=case_id)

    path = card_path(case_id)
    if not path.exists():
        return failure("not_found", f"Knowledge card '{case_id}' not found", case_id=case_id)

    try:
        card = load_card_by_id(case_id)
        card.setdefault("effect", {})
        card["effect"]["label"] = effect_label
        card["effect"]["confidence"] = confidence
        card["effect"]["step_improvement"] = step_improvement
        if thought_match is not None:
            card["effect"]["thought_match"] = thought_match
        if metric_changes is not None:
            card["effect"]["metric_changes"] = metric_changes

        card.setdefault("runtime", {})
        if overwrite_runtime_created_at or "updated_at" not in card["runtime"]:
            card["runtime"]["updated_at"] = utc_now_iso()

        is_valid, error_msg = validate_knowledge_card_object(card)
        if not is_valid:
            return failure(
                "schema_validation_failed",
                error_msg or "Schema validation failed after effect update",
                case_id=case_id,
            )

        atomic_json_dump(card, path)
        update_index_with_card(card)
        return success({"case_id": case_id, "message": "Effect updated successfully"}, case_id=case_id)
    except Exception as e:
        logger.exception("Failed to update effect for %s", case_id)
        return failure("update_failed", str(e), case_id=case_id)


@mcp.tool()
def create_knowledge_card_from_diff(
    code_diff_entry: Dict[str, Any],
    session_id: str,
    step_index: int,
    profiling_data: Optional[Dict[str, Any]] = None,
    kernel_info: Optional[Dict[str, Any]] = None,
    scene_info: Optional[Dict[str, Any]] = None,
    runtime_info: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Create and save a knowledge card from code_diff output and optional profiling data.

    This is the preferred write path after code_diff_learning or similar structured
    optimization analysis.
    """
    if not isinstance(code_diff_entry, dict):
        return failure("invalid_input", "code_diff_entry must be a dictionary")

    kernel_name = (
        code_diff_entry.get("kernel_name")
        or maybe_get(kernel_info, "name")
        or "unknown"
    )
    if not kernel_name or kernel_name == "unknown":
        return failure(
            "missing_kernel_name",
            "kernel_name is required in code_diff_entry or kernel_info",
        )

    case_id = build_case_id(kernel_name, session_id, step_index)

    kernel_category = maybe_get(kernel_info, "category") or infer_category(kernel_name)
    kernel_compute_type = (
        maybe_get(profiling_data, "bottleneck_type")
        or maybe_get(kernel_info, "compute_type")
        or "unknown"
    )

    thought_id = code_diff_entry.get("target_idea", "unknown")
    thought_title = generate_thought_title(thought_id, kernel_name)
    thought_description = code_diff_entry.get("diff_summary", "")
    thought_reason = code_diff_entry.get("diff_detail", "")

    change_category = code_diff_entry.get("change_category", "unknown")
    opt_layer = map_category_to_opt_layer(change_category)

    effect_label = "effective"
    effect_confidence = float(maybe_get(profiling_data, "confidence", 0.8))
    thought_match = "high"
    step_improvement = 1.0

    task_duration = maybe_get(profiling_data, "task_duration_us", {})
    task_before = task_duration.get("before", 0) if isinstance(task_duration, dict) else 0
    task_after = task_duration.get("after", task_before) if isinstance(task_duration, dict) else task_before

    if task_before and task_after:
        try:
            step_improvement = round(float(task_before) / float(task_after), 4)
        except Exception:
            step_improvement = 1.0

    if step_improvement > 1.05:
        effect_label = "effective"
    elif step_improvement > 1.01:
        effect_label = "partial"
    elif step_improvement >= 1.0:
        effect_label = "ineffective"
    else:
        effect_label = "negative"

    scene_shape = maybe_get(scene_info, "shape_signature", "N=unknown")
    scene_dtype = maybe_get(scene_info, "dtype", "float32")
    scene_data_tags = maybe_get(scene_info, "data_tags", []) or []
    scene_task_tags = maybe_get(scene_info, "task_tags", ["inference"]) or ["inference"]

    prev_case_id = "baseline" if step_index == 0 else build_case_id(kernel_name, session_id, step_index - 1)
    prev_thought_id = "baseline" if step_index == 0 else f"step_{step_index - 1}"

    runtime_device = maybe_get(runtime_info, "device", "Ascend910B")
    runtime_agent = maybe_get(runtime_info, "agent", "kernel-memory")
    runtime_iteration = maybe_get(runtime_info, "iteration", step_index)
    runtime_dsl_before = maybe_get(
        runtime_info,
        "dsl_before",
        f"output/{kernel_name}/{kernel_name}_dsl.py",
    )
    runtime_dsl_after = maybe_get(
        runtime_info,
        "dsl_after",
        f"output/{kernel_name}/{kernel_name}_optimized_{step_index}.py",
    )
    runtime_profiling_path = maybe_get(
        runtime_info,
        "profiling_path",
        f"output/{kernel_name}/profiling_round_{step_index}.json",
    )

    metric_changes: Dict[str, Any] = {}
    custom_median_ms = maybe_get(profiling_data, "custom_median_ms")
    if isinstance(task_duration, dict):
        metric_changes["task_duration_us"] = task_duration
    if isinstance(custom_median_ms, dict):
        metric_changes["e2e_time_ms"] = custom_median_ms

    knowledge_card = {
        "schema_version": "0.2.0",
        "case_id": case_id,
        "kernel": {
            "name": kernel_name,
            "category": kernel_category,
            "compute_type": kernel_compute_type,
            "path": runtime_dsl_after,
        },
        "thought": {
            "id": thought_id,
            "title": thought_title,
            "description": thought_description,
            "reason": thought_reason,
            "opt_layer": opt_layer,
            "related_opt_layers": [],
            "target_bottleneck": maybe_get(profiling_data, "bottleneck_type", "unknown"),
            "apply_scenario": "performance",
            "apply_stage": "optimization",
        },
        "scene": {
            "shape_signature": scene_shape,
            "dtype": scene_dtype,
            "data_tags": scene_data_tags,
            "task_tags": scene_task_tags,
        },
        "effect": {
            "label": effect_label,
            "confidence": effect_confidence,
            "thought_match": thought_match,
            "step_improvement": step_improvement,
            "metric_changes": metric_changes,
        },
        "chain": {
            "session_id": session_id,
            "step_index": step_index,
            "prev_thought_id": prev_thought_id,
            "prev_case_id": prev_case_id,
            "full_sequence_hint": [f"step_{i}" for i in range(step_index + 1)],
        },
        "runtime": {
            "device": runtime_device,
            "agent": runtime_agent,
            "iteration": runtime_iteration,
            "dsl_before": runtime_dsl_before,
            "dsl_after": runtime_dsl_after,
            "profiling_path": runtime_profiling_path,
            "created_at": utc_now_iso(),
        },
    }

    return save_knowledge_card(knowledge_card, overwrite=overwrite)


@mcp.tool()
def delete_knowledge_card(case_id: str) -> Dict[str, Any]:
    """
    Delete a knowledge card by case_id.

    Use only when deletion is explicitly requested.
    """
    ok, err = validate_case_id(case_id)
    if not ok:
        return failure("invalid_case_id", err or "Invalid case_id", case_id=case_id)

    path = card_path(case_id)
    if not path.exists():
        return failure("not_found", f"Knowledge card '{case_id}' not found", case_id=case_id)

    try:
        path.unlink()
        remove_from_index(case_id)
        return success({"case_id": case_id, "message": "Knowledge card deleted successfully"}, case_id=case_id)
    except Exception as e:
        logger.exception("Failed to delete card %s", case_id)
        return failure("delete_failed", str(e), case_id=case_id)


@mcp.tool()
def list_knowledge_card_ids() -> Dict[str, Any]:
    """
    List all known case_ids from the index.
    """
    index = load_index()
    ids = list(index.get("cases", {}).keys())
    ids.sort()
    return success(ids, count=len(ids))


@mcp.tool()
def get_knowledge_card_count() -> Dict[str, Any]:
    """
    Return total card count and counts by category.
    """
    index = load_index()
    cases = index.get("cases", {})
    by_category: Dict[str, int] = {}
    for summary in cases.values():
        cat = summary.get("category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
    return success({"total": len(cases), "by_category": by_category})


# ------------------------------------------------------------------------------
# Backward-compatible aliases
# ------------------------------------------------------------------------------

@mcp.tool()
def read_knowledge_card(case_id: str) -> Dict[str, Any]:
    """
    Backward-compatible alias of get_knowledge_card_by_id(case_id).
    """
    return get_knowledge_card_by_id(case_id)


@mcp.tool()
def write_knowledge_card(card: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
    """
    Backward-compatible alias of save_knowledge_card(card, overwrite=False).
    """
    return save_knowledge_card(card=card, overwrite=overwrite)


@mcp.tool()
def query_knowledge_cards(
    kernel_name: Optional[str] = None,
    category: Optional[str] = None,
    compute_type: Optional[str] = None,
    effect_label: Optional[str] = None,
    session_id: Optional[str] = None,
    thought_id: Optional[str] = None,
    target_bottleneck: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Backward-compatible alias of search_knowledge_card_summaries(...).

    This alias returns summaries, not full cards, by design.
    """
    return search_knowledge_card_summaries(
        kernel_name=kernel_name,
        category=category,
        compute_type=compute_type,
        effect_label=effect_label,
        session_id=session_id,
        thought_id=thought_id,
        target_bottleneck=target_bottleneck,
        limit=limit,
    )


@mcp.tool()
def list_knowledge_cards() -> Dict[str, Any]:
    """
    Backward-compatible alias of list_knowledge_card_ids().
    """
    return list_knowledge_card_ids()


@mcp.tool()
def validate_schema(card: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward-compatible alias of validate_knowledge_card(card).
    """
    return validate_knowledge_card(card)


@mcp.tool()
def get_schema() -> Dict[str, Any]:
    """
    Backward-compatible alias of get_knowledge_card_schema().
    """
    return get_knowledge_card_schema()


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------

def main() -> None:
    """MCP server entry point (stdio transport)."""
    logger.info("Starting Knowledge Card MCP Server")
    logger.info("Base directory: %s", BASE_DIR)
    logger.info("Records directory: %s", RECORDS_DIR)
    logger.info("Cases directory: %s", CASES_DIR)
    logger.info("Schema file: %s", SCHEMA_FILE)
    mcp.run()


if __name__ == "__main__":
    main()