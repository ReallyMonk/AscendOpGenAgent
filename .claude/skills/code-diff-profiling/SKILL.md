---
name: code-diff-profiling
description: Analyze code changes related to profiling and performance measurement, record optimization insights from profiling data
---

## What I do

Analyze **code changes** (before vs after profiling-guided optimization) to extract **profiling-driven optimization insights**. This skill focuses on understanding how profiling data leads to specific code modifications and records these patterns for future optimization.

---

## When to use me

Use this skill when you have:

- **Profiling data** (timing, hardware counters, bottleneck reports)
- **Code changes** made based on profiling analysis
- Want to **validate** that the changes address the identified bottlenecks
- Want to **accumulate** profiling-to-optimization patterns

---

## Workflow

### Step 1: Input Collection

- Receive **Original Code** (before profiling-guided optimization)
- Receive **Optimized Code** (after applying profiling-based changes)
- Receive **Profiling Report** (bottleneck analysis, timing data, hardware metrics)
- Receive **Bottleneck Round Number** (e.g., `bottleneck_round_1.md`)

### Step 2: Bottleneck Analysis

- Read the profiling bottleneck report
- Identify the specific bottlenecks that were targeted:
  - Memory bottleneck (GM/L2/L1/UB access)
  - Compute bottleneck (arithmetic intensity)
  - Pipeline bottleneck (data flow, synchronization)
  - Control flow bottleneck (branching, irregular access)

### Step 3: Code Change Analysis

For each identified bottleneck:
- Analyze how the code was modified to address it
- Determine the optimization strategy used:
  - Tiling/blocking for memory optimization
  - Vectorization for compute optimization
  - Pipelining for data flow optimization
  - Loop restructuring for control flow optimization

### Step 4: Profiling-Optimization Mapping

Create a mapping between:
- **Profiling metric** → **Code change** → **Optimization strategy**
- **Bottleneck type** → **Solution pattern**

### Step 5: Knowledge Card Storage (MCP)

Call the MCP tool to store the knowledge card:
```
knowledge_cards_create_knowledge_card_from_diff
```

**Required parameters:**
- `code_diff_entry`: { kernel_name, target_idea, change_category, diff_summary, diff_detail }
- `session_id`: Unique session ID (e.g., `sess_{op}_{YYYYMMDD}_{seq}`)
- `step_index`: Current step index

**Optional parameters:**
- `profiling_data`: { bottleneck_type, task_duration_us: {before, after}, confidence }
- `kernel_info`: { category, compute_type }
- `scene_info`: { shape_signature, dtype, data_tags, task_tags }
- `runtime_info`: { device, dsl_before, dsl_after, profiling_path }

If no meaningful changes are found, no knowledge card is created.

---

## Requirements & Constraints

### Bottleneck Categories

#### Memory Bottleneck
- High GM/L2/L1/UB access
- Poor data reuse
- Suboptimal data layout
- Alignment issues

#### Compute Bottleneck
- Low arithmetic intensity
- Redundant computations
- Suboptimal data types

#### Pipeline Bottleneck
- Data dependency stalls
- Synchronization overhead
- Load imbalance

#### Control Flow Bottleneck
- Branch misprediction
- Irregular memory access
- Dynamic loop bounds

### Code Change Patterns to Identify

| Bottleneck Type | Common Optimization Patterns |
|----------------|----------------------------|
| Memory | Tiling, blocking, prefetching, data layout change |
| Compute | Vectorization, algorithm change, reduced precision |
| Pipeline | Loop fusion, async copy, double buffering |
| Control | Loop reordering, branch elimination, scatter/gather |

### Kernel Naming Rule

Same as code-diff-learning:
- Use canonical operator name
- Do NOT include suffixes like `Custom`, `Impl`, `Test`, `AutoGen`

---

## Input

### Required
- **Original Code** – Code before profiling-guided optimization
- **Optimized Code** – Code after profiling-based changes
- **Profiling Report** – Bottleneck analysis (e.g., `bottleneck_round_{n}.md`)

### Optional
- **kernel_name** – Canonical operator name
- **iteration** – Optimization iteration number
- **target_hardware** – Hardware target (e.g., Ascend 910B)

---

## Output

### Knowledge Card Creation

When profiling-guided optimizations are applied, call the MCP tool to store the knowledge card:

```
knowledge_cards_create_knowledge_card_from_diff
```

**Required parameters:**
- `code_diff_entry`: { kernel_name, target_idea, change_category, diff_summary, diff_detail }
- `session_id`: Unique session ID (e.g., `sess_{op}_{YYYYMMDD}_{seq}`)
- `step_index`: Current step index (0=baseline, 1,2,...=optimization steps)

**Optional parameters:**
- `profiling_data`: { bottleneck_type, task_duration_us: {before, after}, confidence }
- `kernel_info`: { category, compute_type }
- `scene_info`: { shape_signature, dtype, data_tags, task_tags }
- `runtime_info`: { device, dsl_before, dsl_after, profiling_path }

**Output:** The returned knowledge card includes: case_id, kernel, thought, scene, effect, chain, runtime.

If no meaningful changes are found, no knowledge card is created.
