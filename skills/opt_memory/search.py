"""ReMe 封装：语义检索 + 结构化字段过滤。

两阶段检索：
1. 结构化过滤 — 按 operator / thought_id / label / opt_layer 在 Case 索引中精确匹配
2. 语义检索 — 调用 ReMeLight.memory_search（BM25 + 向量混合）做模糊匹配

最终结果合并去重输出。
"""

from .case_manager import list_cases
from .common import get_reme


def search(
    *,
    query: str,
    operator: str | None = None,
    thought_id: str | None = None,
    label: str | None = None,
    opt_layer: str | None = None,
    max_results: int = 5,
    working_dir: str = ".",
) -> dict:
    """语义检索历史优化经验。

    先做结构化过滤（Case 索引精确匹配），再做 ReMe 语义检索，
    结果合并去重后返回。

    Returns:
        dict: {"results": list[dict], "query": str}
    """
    results: list[dict] = []

    # 第一阶段：结构化过滤
    structured = list_cases(
        operator=operator,
        thought_id=thought_id,
        label=label,
        opt_layer=opt_layer,
        working_dir=working_dir,
    )
    for s in structured:
        s["_source"] = "structured"
        results.append(s)

    # 第二阶段：ReMe 语义检索
    try:
        reme = get_reme(working_dir)
        reme_results = reme.memory_search(query=query, max_results=max_results)
        for r in reme_results:
            r["_source"] = "semantic"
            # 去重：如果 case_id 已在结构化结果中，跳过
            case_id = r.get("case_id", "")
            if not case_id or not any(s.get("case_id") == case_id for s in structured):
                results.append(r)
    except Exception:
        # ReMe 不可用时静默降级，只返回结构化结果
        pass

    # 截断到 max_results
    results = results[:max_results]

    return {"results": results, "query": query}
