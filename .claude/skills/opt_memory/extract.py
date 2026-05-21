"""模式提取器（Phase 3 骨架）。

从优化日志和 Case 中自动提取可复用的优化模式，
写入 opt_memory/patterns/{pattern_name}.md。

当前版本：手动创建模式文件，自动提取待 Phase 3 实现。
"""

from pathlib import Path

from .common import ensure_dir, get_opt_dir


def add_pattern(
    *,
    name: str,
    core_idea: str,
    operator_types: str = "",
    shape_range: str = "",
    hardware_constraints: str = "",
    code_skeleton: str = "",
    limitations: str = "",
    source_operator: str = "",
    source_date: str = "",
    related_cases: str = "",
    working_dir: str = ".",
) -> dict:
    """手动添加一条优化模式。

    Returns:
        dict: {"name": str, "path": str}
    """
    patterns_dir = ensure_dir(get_opt_dir(working_dir) / "patterns")
    pattern_path = patterns_dir / f"{name}.md"

    content = f"""# {name}

## 适用场景
- **算子类型**：{operator_types}
- **Shape 范围**：{shape_range}
- **硬件约束**：{hardware_constraints}

## 核心思路
{core_idea}

## 代码模板
```python
{code_skeleton}
```

## 已知限制
- {limitations}

## 来源
- 首次发现于：{source_operator} 优化 {source_date}
- 相关 Case：{related_cases}

## 状态
待验证
"""
    with open(pattern_path, "w") as f:
        f.write(content)

    return {"name": name, "path": str(pattern_path)}


def list_patterns(working_dir: str = ".") -> dict:
    """列出所有已有优化模式。

    Returns:
        dict: {"patterns": list[dict]}
    """
    patterns_dir = get_opt_dir(working_dir) / "patterns"
    if not patterns_dir.exists():
        return {"patterns": []}

    patterns = []
    for md_file in sorted(patterns_dir.glob("*.md")):
        patterns.append({
            "name": md_file.stem,
            "path": str(md_file),
        })

    return {"patterns": patterns}
